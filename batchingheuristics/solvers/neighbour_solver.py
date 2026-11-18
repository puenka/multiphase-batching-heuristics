from batchingheuristics import Instance, Batch, Order, WarehouseItem, Article
from batchingheuristics.solvers.picklist_computation import cvrp_picklists, heuristic_picklists
import random

from collections import defaultdict
from loguru import logger

from batchingheuristics.solvers.simple_solvers import PhaseSolver


class NeighbourAisleSolver(PhaseSolver):
    """
    This solver extends the RDGA approach. Instead of randomly sampling new orders picking the best fit in every step, a
    list of candidates orders is maintained and updated in every step, by adding new candidate orders that are fulfilled
    by items that are positioned in the same aisle as a previously picked item.
    """

    def __init__(self, instance: Instance, n_candidates=100, new_per_item=10):
        """

        Args:
            instance (Instance):
            n_candidates (int, optional): The size of the candidate order list. Defaults to 100.
            new_per_item (int, optional): The number of new orders to be considered for every item. Defaults to 10.
        """
        self.instance = instance
        self.n_candidates = n_candidates
        self.new_per_item = new_per_item

        self.items_by_article = self.instance.get_article_item_dict()
        self.items_by_zone = self.instance.get_zone_items_dict()

    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        remaining_item_goal = self.instance.parameters.min_number_requested_items
        remaining_orders = list(self.instance.orders.copy())
        remaining_orders_by_article = defaultdict(set)
        for order in self.instance.orders:
            for article in order.positions:
                remaining_orders_by_article[article].add(order)

        batches = []

        while remaining_item_goal > 0 and len(remaining_orders) > 0:
            # new batch
            batch_orders = []
            batch_items = []
            candidate_orders = random.sample(remaining_orders, k=self.n_candidates)

            # fill batch up to maximum order limit
            while len(batch_orders) < self.instance.parameters.max_orders_per_batch:
                # pick the best order from the current candidates
                new_order, new_items = self._compute_next_order(candidate_orders, batch_items)
                batch_orders.append(new_order)
                remaining_orders.remove(new_order)
                candidate_orders.remove(new_order)
                remaining_item_goal -= len(new_order.positions)
                batch_items.extend(new_items)
                for item in new_items:
                    self.items_by_article[item.article].remove(item)
                    self.items_by_zone[item.zone].remove(item)
                    if (
                        new_order in remaining_orders_by_article[item.article]
                    ):  # if an order picks the same article twice, it could already be removed
                        remaining_orders_by_article[item.article].remove(new_order)
                if len(remaining_orders) == 0:
                    break
                # augment/change candidate order list by adding relevant neighbouring items -> articles -> orders

                neighbour_count = 0
                for item in new_items:
                    neighbours = self._neighbouring_orders_aisle(
                        remaining_orders_by_article, candidate_orders, item, self.new_per_item
                    )
                    candidate_orders.extend(neighbours)
                    neighbour_count += len(neighbours)
                if len(candidate_orders) > self.n_candidates:
                    candidate_orders = random.sample(candidate_orders, k=self.n_candidates)
                else:
                    candidate_orders.extend(
                        random.sample(
                            [order for order in remaining_orders if order not in candidate_orders],
                            k=self.n_candidates - len(candidate_orders),
                        )
                    )

            logger.debug(f"Batch with {len(batch_orders)} and {len(batch_items)} items created.")
            if cvrp_routing:
                picklists = cvrp_picklists(self.instance, batch_items)
            else:
                picklists = heuristic_picklists(batch_items, self.instance.parameters.max_container_volume)
            batches.append(Batch(batch_orders, picklists))

        return batches

    def _compute_next_order(
        self, candidate_orders: list[Order], picked_items: list[WarehouseItem]
    ) -> tuple[Order, list[WarehouseItem]]:
        selected_items_by_zone = {zone: [WarehouseItem("conveyor", None, 0, 0, zone)] for zone in self.instance.zones}
        for i in picked_items:
            selected_items_by_zone[i.zone].append(i)

        def distance_per_item(order: Order) -> tuple[float, list[WarehouseItem]]:
            total_distance = 0
            add_items = set()
            add_items_by_zone = defaultdict(list)
            for article in order.positions:
                distance, item = min(
                    (
                        min(
                            self.instance.distance(item1, item2)
                            for item2 in selected_items_by_zone[item1.zone] + add_items_by_zone[item1.zone]
                        ),
                        item1,
                    )
                    for item1 in self.items_by_article[article]
                    if item1 not in add_items
                )
                total_distance += distance
                add_items_by_zone[item.zone].append(item)
                add_items.add(item)
            return total_distance / len(add_items), list(add_items)

        _dist, items, order = min((*distance_per_item(order), order) for order in candidate_orders)
        return order, items

    def _neighbouring_orders_aisle(
        self, orders_by_article: dict[Article, set[Order]], candidate_orders, item: WarehouseItem, k: int
    ) -> list[Order]:
        """
        Computes a (random) selection of `k` orders that have at least one article fulfilled by an item in the same
        aisle at the same side of the cross aisle of the given `item`.
        """
        close_items = [i for i in self.items_by_zone[item.zone] if i.aisle == item.aisle and i.row * item.row > 0]
        potential_articles = [i.article for i in close_items]
        potential_orders = set()
        for article in potential_articles:
            potential_orders.update(orders_by_article[article])
        for order in candidate_orders:
            if order in potential_orders:
                potential_orders.remove(order)
        if len(potential_orders) <= k:
            return list(potential_orders)
        else:
            return random.sample(list(potential_orders), k=k)


class NeighbourSolver:
    """
    Similar to the `NeighbourAisleSolver`, but instead of all items in the same aisle, nearby items are considered.
    """

    def __init__(self, instance: Instance, n_candidates=100, new_per_item=10, max_dist=15):
        """_summary_

        Args:
            instance (Instance):
            n_candidates (int, optional): The size of the candidate order list. Defaults to 100.
            new_per_item (int, optional): The number of new orders to be considered for every item. Defaults to 10.
            max_dist (int, optional): All items within this distance are considered for new orders. Defaults to 15.
        """
        self.instance = instance
        self.n_candidates = n_candidates
        self.new_per_item = new_per_item
        self.max_dist = max_dist

        self.items_by_article = self.instance.get_article_item_dict()
        self.items_by_zone = self.instance.get_zone_items_dict()

    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        remaining_item_goal = self.instance.parameters.min_number_requested_items
        remaining_orders = list(self.instance.orders.copy())
        remaining_orders_by_article = defaultdict(set)
        for order in self.instance.orders:
            for article in order.positions:
                remaining_orders_by_article[article].add(order)

        batches = []

        while remaining_item_goal > 0 and len(remaining_orders) > 0:
            # new batch
            batch_orders = []
            batch_items = []
            candidate_orders = random.sample(remaining_orders, k=self.n_candidates)

            # fill batch up to maximum order limit
            while len(batch_orders) < self.instance.parameters.max_orders_per_batch:
                # pick the best order from the current candidates
                new_order, new_items = self._compute_next_order(candidate_orders, batch_items)
                batch_orders.append(new_order)
                remaining_orders.remove(new_order)
                candidate_orders.remove(new_order)
                remaining_item_goal -= len(new_order.positions)
                batch_items.extend(new_items)
                for item in new_items:
                    self.items_by_article[item.article].remove(item)
                    self.items_by_zone[item.zone].remove(item)
                    if (
                        new_order in remaining_orders_by_article[item.article]
                    ):  # if an order picks the same article twice, it could already be removed
                        remaining_orders_by_article[item.article].remove(new_order)
                if len(remaining_orders) == 0:
                    break
                # augment/change candidate order list by adding relevant neighbouring items -> articles -> orders

                neighbour_count = 0
                for item in new_items:
                    neighbours = self._neighbouring_orders_dist(
                        remaining_orders_by_article, candidate_orders, item, self.new_per_item, self.max_dist
                    )
                    candidate_orders.extend(neighbours)
                    neighbour_count += len(neighbours)
                if len(candidate_orders) > self.n_candidates:
                    candidate_orders = random.sample(candidate_orders, k=self.n_candidates)
                else:
                    candidate_orders.extend(
                        random.sample(
                            [order for order in remaining_orders if order not in candidate_orders],
                            k=self.n_candidates - len(candidate_orders),
                        )
                    )

            logger.debug(f"Batch with {len(batch_orders)} and {len(batch_items)} items created.")
            if cvrp_routing:
                picklists = cvrp_picklists(self.instance, batch_items)
            else:
                picklists = heuristic_picklists(batch_items, self.instance.parameters.max_container_volume)
            batches.append(Batch(batch_orders, picklists))

        return batches

    def _compute_next_order(
        self, candidate_orders: list[Order], picked_items: list[WarehouseItem]
    ) -> tuple[Order, list[WarehouseItem]]:
        selected_items_by_zone = {zone: [WarehouseItem("conveyor", None, 0, 0, zone)] for zone in self.instance.zones}
        for i in picked_items:
            selected_items_by_zone[i.zone].append(i)

        def distance_per_item(order: Order) -> tuple[float, list[WarehouseItem]]:
            total_distance = 0
            add_items = set()
            add_items_by_zone = defaultdict(list)
            for article in order.positions:
                distance, item = min(
                    (
                        min(
                            self.instance.distance(item1, item2)
                            for item2 in selected_items_by_zone[item1.zone] + add_items_by_zone[item1.zone]
                        ),
                        item1,
                    )
                    for item1 in self.items_by_article[article]
                    if item1 not in add_items
                )
                total_distance += distance
                add_items_by_zone[item.zone].append(item)
                add_items.add(item)
            return total_distance / len(add_items), list(add_items)

        _dist, items, order = min((*distance_per_item(order), order) for order in candidate_orders)
        return order, items

    def _neighbouring_orders_dist(
        self,
        orders_by_article: dict[Article, set[Order]],
        candidate_orders,
        item: WarehouseItem,
        k: int,
        max_dist: float,
    ) -> list[Order]:
        """
        Computes a (random) selection of `k` orders that have at least one article fulfilled by an item within
        `max_dist` of the given `item`.
        """
        close_items = [i for i in self.items_by_zone[item.zone] if self.instance.distance(item, i) < max_dist]
        potential_articles = [i.article for i in close_items]
        potential_orders = set()
        for article in potential_articles:
            potential_orders.update(orders_by_article[article])
        for order in candidate_orders:
            if order in potential_orders:
                potential_orders.remove(order)
        if len(potential_orders) <= k:
            return list(potential_orders)
        else:
            return random.sample(list(potential_orders), k=k)
