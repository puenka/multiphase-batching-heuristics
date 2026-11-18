import math
from collections import defaultdict, Counter
from loguru import logger
from bisect import bisect

from batchingheuristics import Instance, Batch, WarehouseItem, Article
from batchingheuristics.solvers.picklist_computation import (
    cvrp_picklists,
    heuristic_picklists,
)


class ZGASolver:
    """
    Greedily choose orders to maximize the number of items in the same zone/minimizing how many zones are being used.
    Conceptually, we use the sum of some objective function `zonecount_mapping` that is locally optimized for the order
    choice. The function takes the current number of items picked from any zone. By default, it uses `x -> x**2`. If
    `zonecount_mapping` is an integer, a special function will be used that aims to put the given number of items into
    each zone.
    For each greedy choice, we pick the best marginal increase in this function.
    Additionally, the concept of a "zone" can be generalized to any function `zone_id` returning an identifier for each
    item. Items with the same identifier are considered to be in the same "zone".

    The parameter `zone_id` should map items to some identifier (str). Items with the same identifier are considered
    to be in the same zone. The second parameter `zonecount_mapping` should be a function `int -> int`, taking the
    current number of items in a zone and returning the corresponding objective value for this number.
    """

    def __init__(self, instance, zone_id=None, zonecount_mapping=None):
        self.instance = instance
        if zone_id is None:
            self.zone_id = lambda item: item.zone
        else:
            self.zone_id = zone_id

        if zonecount_mapping is None:  # use quadratic function as default
            self.zonecount_mapping = lambda x: x**2
        elif isinstance(zonecount_mapping, int):
            target_count = zonecount_mapping
            self.zonecount_mapping = lambda x: (
                10 * (1 - math.exp(-4 * x / target_count))
            )

    def compute_batches(self, cvrp_routing=False) -> list[Batch]:
        def obj_marginal_gains(n):
            return self.zonecount_mapping(n + 1) - self.zonecount_mapping(n)

        item_goal = (
            self.instance.parameters.min_number_requested_items
        )  # running number of times to assign
        remaining_orders = set(self.instance.orders.copy())

        warehouse_article_items: dict[Article, set[WarehouseItem]] = defaultdict(set)
        for item in self.instance.warehouse_items:
            warehouse_article_items[item.article].add(item)

        batches = []

        # repeat while item goal is not reached and there are remaining orders in the pool
        while item_goal > 0 and len(remaining_orders) > 0:
            logger.debug(f"Remaining items to batch: {item_goal}")
            batch_orders = []
            batch_selected_items = []
            logger.debug(
                f"Creating a batch, total remaining orders in the pool: {len(remaining_orders)}"
            )

            zone_count = Counter()

            while (
                len(remaining_orders) > 0
                and len(batch_orders) < self.instance.parameters.max_orders_per_batch
            ):
                # compute next best order to be selected
                selected_order = None
                selected_items = []
                selected_avg_marginal_gain = -1

                for order in remaining_orders:
                    candidate_order_items = []
                    candidate_order_marginal_gains = 0
                    for article in order.positions:
                        candidate_article_items = [
                            item
                            for item in warehouse_article_items[article]
                            if item not in candidate_order_items
                        ]
                        best_item = max(
                            candidate_article_items,
                            key=lambda item: obj_marginal_gains(
                                zone_count[self.zone_id(item)]
                            ),
                        )
                        candidate_order_items.append(best_item)
                        candidate_order_marginal_gains += obj_marginal_gains(
                            zone_count[self.zone_id(best_item)]
                        )
                    candidate_order_marginal_gains /= len(order.positions)
                    if selected_avg_marginal_gain < candidate_order_marginal_gains:
                        selected_order = order
                        selected_items = candidate_order_items
                        selected_avg_marginal_gain = candidate_order_marginal_gains

                batch_orders.append(selected_order)
                batch_selected_items.extend(selected_items)
                for item in selected_items:
                    zone_count[self.zone_id(item)] += 1
                    warehouse_article_items[item.article].remove(item)
                remaining_orders.remove(selected_order)

            logger.debug(f"Zone count of current batch: {zone_count}")
            if cvrp_routing:
                batch = Batch(
                    batch_orders,
                    cvrp_picklists(self.instance, batch_selected_items),
                )
            else:
                batch = Batch(
                    batch_orders,
                    heuristic_picklists(
                        batch_selected_items,
                        self.instance.parameters.max_container_volume,
                    ),
                )
            batches.append(batch)
            logger.debug(
                f"Batch created with a total of {len(batch_selected_items)} items"
            )

            item_goal -= len(batch_selected_items)

        return batches


class ZGAQuadSolver(ZGASolver):
    """
    ZGASolver using a separate id for each quadrant of each zone (i.e. 4 subzones corresponding to different signs of
    the row and column number of each item).
    """

    def __init__(self, instance, zonecount_mapping=None):
        def quadrant_id(item):
            return str(item.zone) + str(item.aisle >= 0) + str(item.row >= 0)

        super().__init__(
            instance, zone_id=quadrant_id, zonecount_mapping=zonecount_mapping
        )


class ZGAFiveByNSolver(ZGASolver):
    """
    ZGASolver that splits each zone in the following way, given an additional parameter `n`:
      - Split the aisles into `n` equally sized sections.
      - Along each main aisle (i.e. along 1/3 of the (half-)aisle width), create subzones wrt the aisle sections.
      - Along each aisle, there remains a center part, which receives its own subzone.
    """

    def __init__(self, instance, n=5, zonecount_mapping=None):
        params = instance.parameters

        def row_section(item):
            """
            Returns an index for the row section within 0..4
            """
            section_width = (params.last_row - params.first_row) / 5
            splits = [params.first_row + i * section_width for i in range(1, 5)]
            return "row_sec" + str(bisect(splits, item.row))

        def aisle_section(item):
            """
            Returns an index for the aisle section within 0, ..., n-1
            """
            splits = [
                params.first_aisle + i * (params.last_aisle - params.first_aisle) / n
                for i in range(1, n)
            ]
            return "aisle_sec" + str(bisect(splits, item.aisle))

        def subzone_id(item):
            id_str = (
                str(item.zone) + "/" + row_section(item) + "/" + aisle_section(item)
            )
            return id_str

        super().__init__(
            instance, zone_id=subzone_id, zonecount_mapping=zonecount_mapping
        )


class ZGARadialSolver(ZGASolver):
    """
    Defines 3 subzones according to distance from the depot. Setting`quad=True` further splits each zone into quadrants.
    """

    def __init__(self, instance: Instance, quad=False, zonecount_mapping=None):
        def zone_id(item):
            dist = (abs(item.row) + abs(item.aisle)) // 30
            if self.quad:
                dir_row = item.row >= 0
                dir_aisle = item.aisle >= 0
                return "z" + str(item.zone) + "d" + str((dist, dir_row, dir_aisle))
            else:
                return "z" + str(item.zone) + "d" + str(dist)

        super().__init__(instance, zone_id=zone_id, zonecount_mapping=zonecount_mapping)
        self.quad = quad
