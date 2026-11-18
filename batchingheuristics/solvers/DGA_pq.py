import time  # noqa: N999
from collections import defaultdict

from loguru import logger

from batchingheuristics import Batch, Instance, Order, WarehouseItem
from batchingheuristics.solvers.picklist_computation import (
    cvrp_picklists,
    heuristic_picklists,
)
from batchingheuristics.solvers.priority_queue import MinPQ
from batchingheuristics.solvers.simple_solvers import PhaseSolver


class DGAPQSolver(PhaseSolver):
    """
    Uses our own heap-based implementation of a priority queue for the distance greedy algorithm.

    The flag `tsp` changes from the routing heuristic to a CVRP model (ortools) to compute optimal routes.
    """

    def __init__(self, instance: Instance, skip_article_distance_check=False):
        self.instance = instance
        self.skip_article_distance_check = skip_article_distance_check

    def compute_batches(
        self,
        cvrp_routing=False,
    ) -> list[Batch]:
        return dga_pq(self.instance, cvrp_routing, self.skip_article_distance_check)


"""
Main Performance Points:
- Affected item list currently only excludes opposite side quadrant, has heavy impact on performance, more opt. poss.
- Order cost calculation is fairly costly, might be able to improve by having some form of pq for items?
"""


def dga_pq(
    instance, cvrp_routing=False, skip_article_distance_check=False
) -> list[Batch]:
    logger.info("Setting up problem instance.")
    remaining_item_goal = instance.parameters.min_number_requested_items
    max_batch_size = instance.parameters.max_orders_per_batch
    batches = []
    batch_orders = []
    batch_items = []
    warehouse_picklist_state = WarehousePicklistState(
        instance, skip_article_distance_check
    )

    logger.info("Starting batching.")

    # Main loop
    while (
        remaining_item_goal > 0
        and len(warehouse_picklist_state.remaining_orders.heap) > 0
    ):
        batch_time_1 = time.time()
        while (
            len(batch_orders) < max_batch_size
            and len(warehouse_picklist_state.remaining_orders.heap) > 0
        ):
            next_order, next_items = (
                warehouse_picklist_state.get_next_order_update_state()
            )
            batch_orders.append(next_order)
            batch_items.extend(next_items)

        if cvrp_routing:
            picklists = cvrp_picklists(instance, batch_items)
        else:
            picklists = heuristic_picklists(
                batch_items, instance.parameters.max_container_volume
            )
        batches.append(Batch(batch_orders, picklists))
        items_covered = len(batch_items)
        remaining_item_goal = remaining_item_goal - items_covered

        batch_time_2 = time.time()
        batch_duration = batch_time_2 - batch_time_1
        logger.debug(
            f"Batch created with {len(batch_orders)} orders, {len(batches[-1].picklists)} picklists, "
            f"containing {items_covered} items. {remaining_item_goal} remaining items to cover. "
            f"Creation took {batch_duration:.2f} seconds"
        )

        n_unique = len({bi.id for bi in batch_items})
        n_items = len(batch_items)
        if n_unique != n_items:
            logger.warning(f"Number unique items: {n_unique}, number items: {n_items}")

        batch_orders = []
        batch_items = []
        warehouse_picklist_state.reset_for_new_batch()

    logger.info(f"Batching completed. {len(batches)} batches created.")

    return batches


# Class representing a zone of the warehouse with the associated current picklist state and cost functions
class PicklistZoneState:
    zone_number: int
    row_count: int
    aisle_count: int
    items_set: set[WarehouseItem]
    items_TL: set[WarehouseItem]
    items_TR: set[WarehouseItem]
    items_BL: set[WarehouseItem]
    items_BR: set[WarehouseItem]

    def __init__(self, zone_number, row_count, aisle_count):
        self.zone_number = zone_number
        self.row_count = row_count
        self.aisle_count = aisle_count
        self.activePicklists = []
        self.items_set = set()
        self.items_TL = set()
        self.items_TR = set()
        self.items_BL = set()
        self.items_BR = set()
        self.new_picked_item_positions = []

    def add_item(self, new_item):
        self.items_set.add(new_item)
        a, r = new_item.aisle, new_item.row
        if a < 0 and r < 0:
            self.items_BL.add(new_item)
        elif a > 0 and r < 0:
            self.items_TL.add(new_item)
        elif a < 0 and r >= 0:
            self.items_BR.add(new_item)
        elif a > 0 and r >= 0:
            self.items_TR.add(new_item)

    def remove_items(self, removed_items):
        self.items_set.difference_update(removed_items)
        self.items_TL.difference_update(removed_items)
        self.items_TR.difference_update(removed_items)
        self.items_BL.difference_update(removed_items)
        self.items_BR.difference_update(removed_items)

    # For given items, returns the items which might potentially get the distance updated by inclusion of these
    def get_relevant_items(self, new_items: list[WarehouseItem]) -> set[WarehouseItem]:  # pyright: ignore[reportReturnType]
        pos_set = {
            (-1 if item.aisle < 0 else 1, -1 if item.row < 0 else 1)
            for item in new_items
        }
        if len(pos_set) == 1:
            # Python can't nicely return elements from singleton sets
            a, r = pos_set.pop()
            if a < 0 and r < 0:
                return set().union(self.items_BL, self.items_BR, self.items_TL)
            elif a > 0 and r < 0:
                return set().union(self.items_TR, self.items_TL, self.items_BL)
            elif a < 0 and r >= 0:
                return set().union(self.items_TR, self.items_BL, self.items_BR)
            elif a > 0 and r >= 0:
                return set().union(self.items_TR, self.items_BR, self.items_TL)
        elif len(pos_set) == 0:
            return set()
        else:
            return self.items_set


# Class representing the current state of the warehouse and of picklists during execution of the dynamic route heuristic
class WarehousePicklistState:
    instance: Instance
    zone_count: int
    aisle_count: int
    row_count: int
    picklist_volume: int
    warehouse_items: set[WarehouseItem]
    order_set: set[Order]

    remaining_orders: MinPQ
    zone_dict: dict[int, PicklistZoneState]
    article_items_dict: dict
    item_cost_dict: dict[str, int | float]
    item_cost_dict_cached: dict[str, int | float]
    order_cost_dict: dict[str, int]
    items_orders_dict: dict[str, list[Order]]

    def __init__(self, instance, skip_article_distance_check=False):
        self.instance = instance
        self.skip_article_distance_check = skip_article_distance_check

        self.warehouse_items = set(instance.warehouse_items)
        self.generate_article_items_dict(self.warehouse_items)

        self.order_set = set(instance.orders)

        self.zone_count = self.determine_number_of_zones()
        self.aisle_count = abs(instance.parameters.first_aisle) + abs(
            instance.parameters.last_aisle
        )
        self.row_count = abs(instance.parameters.first_row) + abs(
            instance.parameters.last_row
        )
        self.picklist_volume = instance.parameters.max_container_volume

        self.generate_item_cost_dict(self.warehouse_items)
        self.item_cost_dict_cached = self.item_cost_dict.copy()
        self.generate_items_orders_dict()
        self.zone_dict = {}
        self.initialize_picklist_zones()

        self.generate_order_cost_and_min_pq()

    # The number of zones is not (explicitly) saved in the instance, so we have to manually retrieve it
    def determine_number_of_zones(self):
        return len({x.zone for x in self.warehouse_items})

    # Creates a PicklistZoneState object for each existing zone and adds it to the dictionary
    def initialize_picklist_zones(self):
        for i in range(self.zone_count):
            self.zone_dict[i] = PicklistZoneState(i, self.aisle_count, self.row_count)
        for item in self.warehouse_items:
            self.zone_dict[item.zone].add_item(item)

    # Resets the PicklistZoneState objects and the item dict for a new batch
    def reset_for_new_batch(self):
        self.item_cost_dict = self.item_cost_dict_cached.copy()
        self.generate_order_cost_and_min_pq()

    def get_next_order_update_state(self):
        _order_cost, next_order = self.remaining_orders.heappop()

        # Collect items included in this order choice
        chosen_items = []
        for article in next_order.positions:
            article_items = self.article_items_dict[article.id]
            # Chooses the min cost item, excludes already picked items

            chosen_items.append(
                min(
                    [item for item in article_items if item not in chosen_items],
                    key=lambda item: (
                        self.item_cost_incremental(item, chosen_items),
                        item.id,
                    ),  # tie-break by id
                )
            )

        # Group chosen items by zone
        chosen_items_by_zone = defaultdict(list)
        for chosen_item in chosen_items:
            chosen_items_by_zone[chosen_item.zone].append(chosen_item)

        # Go through selected items and collect all modified items
        zone_ids = {item.zone for item in chosen_items}

        potential_affected_items = set()
        for zone_id in zone_ids:
            potential_affected_items.update(
                self.zone_dict[zone_id].get_relevant_items(
                    chosen_items_by_zone[zone_id]
                )
            )

        affected_items = []
        for item in potential_affected_items:
            dist_old = self.item_cost_dict[item.id]
            dist_new = min(
                [
                    self.instance.distance(item, chosen_item)
                    for chosen_item in chosen_items_by_zone[item.zone]
                ]
            )
            if dist_new < dist_old:
                affected_items.append(item)
                self.item_cost_dict[item.id] = dist_new

        # Collect affected orders
        affected_orders = set()
        for item in affected_items + chosen_items:
            affected_orders.update(self.items_orders_dict[item.id])

        affected_orders.difference_update({next_order})

        # Remove order from items_orders_dict for all relevant items
        for article in next_order.positions:
            for item in self.article_items_dict[article.id]:
                self.items_orders_dict[item.id].remove(next_order)

        # Remove chosen items from item lists and item lists in zone and other data structures
        self.warehouse_items.difference_update(chosen_items)
        for zone_id in zone_ids:
            self.zone_dict[zone_id].remove_items(chosen_items)
        for item in chosen_items:
            temp = self.article_items_dict[item.article.id]
            if item in temp:
                temp.remove(item)
            else:
                raise RuntimeError(f"Item {item} has been picked twice.")
            self.item_cost_dict.pop(item.id, None)

        # Remove order from order list
        self.order_set.remove(next_order)

        # Iterate over orders and update if necessary
        for order in affected_orders:
            old_dist = self.order_cost_dict[order.id]
            new_dist = self.calculate_order_cost(order)
            if old_dist != new_dist:
                self.remaining_orders.update_item((old_dist, order), (new_dist, order))
                self.order_cost_dict[order.id] = new_dist

        return next_order, chosen_items

    # Generates a lookup-table for the warehouse items cost, initialized to distance from origin
    def generate_item_cost_dict(self, warehouse_items: set[WarehouseItem]):
        self.item_cost_dict = {}
        for item in warehouse_items:
            self.item_cost_dict[item.id] = abs(item.aisle) + abs(item.row)

    # Generates a lookup-table for the warehouse items based on article-id
    def generate_article_items_dict(self, warehouse_items: set[WarehouseItem]):
        self.article_items_dict = defaultdict(list)
        for wi in warehouse_items:
            self.article_items_dict[wi.article.id].append(wi)

    # Generates and initializes the min heap containing the (order_cost, order) tuples
    def generate_order_cost_and_min_pq(self):
        cost_order_list = [
            (self.calculate_order_cost(order), order) for order in self.order_set
        ]
        self.order_cost_dict = {i[1].id: i[0] for i in cost_order_list}
        self.remaining_orders = MinPQ(cost_order_list)

    def item_cost_incremental(
        self, item: WarehouseItem, new_items: list[WarehouseItem]
    ):
        dist_ref = self.item_cost_dict[item.id]
        if len(new_items) == 0:
            return dist_ref
        dist_new = min(
            [self.instance.distance(item, new_item) for new_item in new_items]
        )
        return min(dist_ref, dist_new)

    # Calculates the (relative) cost of an order
    def calculate_order_cost(self, order):
        total_cost = 0
        chosen_items = []

        for article in order.positions:
            if self.skip_article_distance_check:
                new_item = min(
                    [
                        item
                        for item in self.article_items_dict[article.id]
                        if item not in chosen_items
                    ],
                    key=lambda item: (
                        self.item_cost_dict[item.id],
                        item.id,
                    ),  # tie-break by id
                )
                new_cost = self.item_cost_dict[new_item.id]
            else:
                new_cost, new_item = min(
                    [  # the second tuple entry ensures consistent tie-breaking towards the lower item id
                        (self.item_cost_incremental(item, chosen_items), item)
                        for item in self.article_items_dict[article.id]
                        if item not in chosen_items
                    ]
                )

            total_cost += new_cost
            chosen_items.append(new_item)
        return (
            round(total_cost / len(order.positions), 3)
            if len(order.positions) > 0
            else 1_000_000
        )

    # Generates a dictionary mapping each item to all orders containing it (i.e. the respective article)
    # Depends on the article_items_dict already being initialized
    def generate_items_orders_dict(self):
        self.items_orders_dict = defaultdict(list)
        for order in self.order_set:
            for art in order.positions:
                art_items = self.article_items_dict[art.id]
                for item in art_items:
                    self.items_orders_dict[item.id].append(order)
