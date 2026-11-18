"""Instead of relying on a heuristic, compute exactly optimal picklists per zone."""

from collections import defaultdict
from batchingheuristics.instances.definitions import WarehouseItem, Instance
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import ceil
from loguru import logger


class CVRPPicklists:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.depots = {
            zone: WarehouseItem("conveyor", None, 0, 0, zone)
            for zone in self.instance.zones
        }

    def compute_picklists(self, items_to_route: list[WarehouseItem]):
        active_zones = {item.zone for item in items_to_route}
        zone_items = {
            zone: [self.depots[zone]]
            + [item for item in items_to_route if item.zone == zone]
            for zone in active_zones
        }

        picklists = []
        for zone in active_zones:
            picklists.extend(self._optimize_zone(zone_items[zone]))

        return picklists

    def _optimize_zone(self, items):
        picklists = []
        depot_idx = 0

        # estimate minimum nr of vehicles by total volume / maximum container volume
        min_num_vehicles = max(
            1,
            ceil(
                sum(
                    [
                        item.article.volume
                        if item.article is not None
                        else 0  # exclude depot
                        for item in items
                    ]
                )
                / self.instance.parameters.max_container_volume
            ),
        )
        solution = None

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.time_limit.seconds = 10

        for num_vehicles in range(min_num_vehicles, len(items) + 1):
            assert (
                num_vehicles < len(items) + 1
            )  # there should be a solution at this point...
            manager = pywrapcp.RoutingIndexManager(len(items), num_vehicles, depot_idx)
            routing = pywrapcp.RoutingModel(manager)

            # distances
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return self.instance.distance(items[from_node], items[to_node])

            transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

            # capacities
            def volume_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return items[from_node].article.volume if from_node > 0 else 0

            volume_callback_idx = routing.RegisterUnaryTransitCallback(volume_callback)
            routing.AddDimension(
                volume_callback_idx,
                0,
                self.instance.parameters.max_container_volume,
                True,
                "Capacity",
            )

            solution = routing.SolveWithParameters(search_params)

            if solution is None:
                logger.debug("No routing solution found, increase number of vehicles.")
            else:
                break

        else:  # if there is no solution with a vehicle per item, something is terribly wrong
            raise RuntimeError("No routing solution has been found.")

        for vehicle_id in range(num_vehicles):
            # initialize route at depot
            vehicle_route = [items[manager.IndexToNode(depot_idx)]]

            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                vehicle_route.append(items[manager.IndexToNode(index)])
            if len(vehicle_route) == 2:
                continue

            picklists.append(vehicle_route[1:-1])  # return routes without depot

        return picklists


def cvrp_picklists(instance: Instance, items: list[WarehouseItem]):
    """
    Given an `instance` and a list of `items`, compute an optimal CVRP solution using ortools.
    The ortools CVRP model will use as few vehicles as possible (i.e. precisely as many as
    necessary to fulfill the requested items per zone).
    """
    logger.debug("Optimize picklists with CVRP solver")

    cvrp = CVRPPicklists(instance)

    return cvrp.compute_picklists(items)


def heuristic_picklists(
    items: list[WarehouseItem], max_container_volume: int
) -> list[list[WarehouseItem]]:
    items_by_zone = defaultdict(list)

    for item in items:
        items_by_zone[item.zone].append(item)

    picklists = []
    for zone in items_by_zone:
        picklist = []
        cur_volume = 0
        for item in sorted(items_by_zone[zone], key=lambda i: (i.aisle, i.row)):
            if cur_volume + item.article.volume <= max_container_volume:
                picklist.append(item)
                cur_volume += item.article.volume
            else:
                picklists.append(picklist)
                picklist = [item]
                cur_volume = item.article.volume
        if len(picklist) > 0:
            picklists.append(picklist)
    return picklists
