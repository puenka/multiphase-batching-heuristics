import math
import random
import time
from collections import defaultdict

import numpy as np
from loguru import logger

from batchingheuristics import Batch, Instance, Order, WarehouseItem
from batchingheuristics.solvers.DGA_pq import WarehousePicklistState
from batchingheuristics.solvers.picklist_computation import (
    cvrp_picklists,
    heuristic_picklists,
)
from batchingheuristics.solvers.priority_queue import MinPQ


class SoftAntiAgingWarehousePicklistState(WarehousePicklistState):
    """
    Adjusts the cost computation in the WarehousePicklistState base class to incorporate order ages.

    The new cost is given by `old_cost * (gamma**age) - alpha*age`.
    Thus `gamma` represents geometric reduction of the costs and `alpha` linear reduction.
    """

    def __init__(self, instance, order_ages, alpha, gamma):
        self.order_ages = order_ages
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(instance)

    def calculate_order_cost(self, order):
        """Reimplement the order cost computation by adjusting the internal cost to change by order age."""
        total_cost = 0
        # Check for duplicates
        if len({article.id for article in order.positions}) == len(order.positions):
            for article in order.positions:
                total_cost += min(
                    [
                        self.item_cost_dict[item.id]
                        for item in self.article_items_dict[article.id]
                    ]
                )
        else:  # There are duplicated articles
            picked_items = []
            for article in order.positions:
                new_item, cost = min(
                    [
                        (i, self.item_cost_dict[i.id])
                        for i in self.article_items_dict[article.id]
                    ],
                    key=lambda x: x[1] if x[0] not in picked_items else 100_000_000,
                )
                picked_items.append(new_item)
                total_cost += cost

        total_cost = (
            total_cost * (self.gamma ** self.order_ages[order])
            - self.alpha * self.order_ages[order]
        )
        return (
            total_cost / len(order.positions) if len(order.positions) > 0 else 1_000_000
        )


class SoftAntiAgingDGAPQSolver:
    def __init__(self, instance, order_ages, alpha=0, gamma=1):
        self.instance = instance
        self.order_ages = order_ages
        self.alpha = alpha
        self.gamma = gamma

    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        logger.info(
            f"Soft Anti Aging DGAPQ with parameters alpha={self.alpha}, gamma={self.gamma}"
        )
        logger.info("Setting up problem instance.")
        remaining_item_goal = self.instance.parameters.min_number_requested_items
        max_batch_size = self.instance.parameters.max_orders_per_batch
        batches = []
        batch_orders = []
        batch_items = []
        warehouse_picklist_state = SoftAntiAgingWarehousePicklistState(
            self.instance, self.order_ages, self.alpha, self.gamma
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
                picklists = cvrp_picklists(self.instance, batch_items)
            else:
                picklists = heuristic_picklists(
                    batch_items, self.instance.parameters.max_container_volume
                )
            batches.append(Batch(batch_orders, picklists))
            items_covered = len(batch_items)
            remaining_item_goal = remaining_item_goal - items_covered

            batch_time_2 = time.time()
            batch_duration = batch_time_2 - batch_time_1
            logger.info(
                f"Batch created with {len(batch_orders)} orders,{len(batches[-1].picklists)} picklists, "
                f"containing {items_covered} items. {remaining_item_goal} remaining items to cover. "
                f"Creation took {batch_duration} seconds."
            )

            logger.info(
                f"Number unique items: {len({bi.id for bi in batch_items})}, number items: {len(batch_items)}"
            )

            batch_orders = []
            batch_items = []
            warehouse_picklist_state.reset_for_new_batch()
        logger.info("Batching completed.")

        return batches


class HardAntiAgingWarehousePicklistState(WarehousePicklistState):
    """
    Adjusts the order selection in the WarehousePicklistState base class to return orders according to age.
    """

    def __init__(self, instance, order_ages, mandatory_age_completion):
        self.order_ages = order_ages
        self.mandatory_age_completion = mandatory_age_completion
        super().__init__(instance)

        self.orders_by_age = defaultdict(list)
        for order in self.order_set:
            self.orders_by_age[order_ages[order]].append(order)

        # populate mandatory order completion list by sampling for each age class
        self.mandatory_orders = []
        for age in self.orders_by_age:
            # pick the required number of orders
            n_of_age = max(
                0, len(self.orders_by_age[age]) - mandatory_age_completion(age)
            )
            self.mandatory_orders.extend(
                random.sample(self.orders_by_age[age], k=n_of_age)
            )

        logger.info(f"{len(self.mandatory_orders)} mandatory orders selected.")

        avg_order_size = np.mean([len(order.positions) for order in self.order_set])
        estimated_batch_count = math.ceil(
            instance.parameters.min_number_requested_items
            / (avg_order_size * instance.parameters.max_orders_per_batch)
        )
        self.n_mandatory_per_batch = math.ceil(
            len(self.mandatory_orders) / estimated_batch_count
        )

        if self.n_mandatory_per_batch > instance.parameters.max_orders_per_batch:
            self.n_mandatory_per_batch = instance.parameters.max_orders_per_batch
            estimated_batch_count = math.ceil(
                len(self.mandatory_orders) / self.n_mandatory_per_batch
            )

        logger.info(
            f"Estimated {estimated_batch_count} batches with {self.n_mandatory_per_batch} mandatory orders each"
        )

        self.mandatory_orders_pq = MinPQ(
            [
                (self.calculate_order_cost(order), order)
                for order in self.mandatory_orders
            ]
        )
        self.batch_order_counter = 0
        self.batch_mandatory_phase = True

    def reset_for_new_batch(self):
        super().reset_for_new_batch()
        self.mandatory_orders_pq = MinPQ(
            [
                (self.calculate_order_cost(order), order)
                for order in self.mandatory_orders
            ]
        )
        self.batch_order_counter = 0
        self.batch_mandatory_phase = True

    def get_next_order_update_state(
        self,
    ) -> tuple[Order | None, list[WarehouseItem] | None]:

        if self.batch_mandatory_phase and (
            self.batch_order_counter >= self.n_mandatory_per_batch
            or len(self.mandatory_orders) == 0
        ):
            logger.debug("Mandatory phase completed. Now batching any order.")
            self.batch_mandatory_phase = False
            super().reset_for_new_batch()

        # Pick the mandatory orders for the first items in each batch, only then pick from the full list
        if self.batch_mandatory_phase:
            _order_cost, next_order = self.mandatory_orders_pq.heappop()

            self.mandatory_orders.remove(next_order)
        else:
            if len(self.remaining_orders) == 0:
                return None, None
            _order_cost, next_order = self.remaining_orders.heappop()

            if next_order in self.mandatory_orders:
                self.mandatory_orders.remove(next_order)
                # mandatory_orders_pq does not need to be updated, as it will only be used after resetting it
        self.batch_order_counter += 1

        # Collect items included in this order choice
        chosen_items = []
        for article in next_order.positions:
            article_items = self.article_items_dict[article.id]
            # Chooses the min cost item, excludes already picked items
            chosen_items.append(
                min(
                    article_items,
                    key=lambda item: (
                        self.item_cost_dict[item.id]
                        if item not in chosen_items
                        else 100_000_000
                    ),
                )
            )

        # Code for checking if the picked articles contains a duplicated article
        if len({x.id for x in next_order.positions}) != len(next_order.positions):
            logger.warning(
                f"! {[art.id for art in next_order.positions]} - {[ci.id for ci in chosen_items]}"
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
        potential_affected_items.difference_update(chosen_items)

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
        for item in affected_items:
            affected_orders.update(self.items_orders_dict[item.id])

        affected_orders.difference_update({next_order})
        if self.batch_mandatory_phase:
            affected_orders.intersection_update(self.mandatory_orders)

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
                logger.warning("Duplicated item picked. ")
            self.item_cost_dict.pop(item.id, None)

        # Remove order from order list
        self.order_set.remove(next_order)

        # Iterate over orders and update if necessary
        for order in affected_orders:
            old_dist = self.order_cost_dict[order.id]
            new_dist = self.calculate_order_cost(order)
            if old_dist != new_dist:
                # If we're in the first stage, update the mandatory order heap as well
                if self.batch_mandatory_phase:
                    self.mandatory_orders_pq.update_item(
                        (old_dist, order), (new_dist, order)
                    )
                else:
                    self.remaining_orders.update_item(
                        (old_dist, order), (new_dist, order)
                    )
                self.order_cost_dict[order.id] = new_dist

        return next_order, chosen_items


class HardAntiAgingDGAPQSolver:
    def __init__(
        self,
        instance,
        order_ages: dict[Order, int],
        mandatory_age_completion: list[int],
    ):
        """
        `mandatory_age_completion` should be a list that takes an integer number (some order age) and returns the
        maximum number of orders of the given age that may still remain after the current phase is over. All unspecified
        ages are assumed to have a limit of 0.

        The solver then randomly samples orders from each age group that are set to be completed within the current
        phase, regardless of their cost. Those mandatory orders are then distributed onto the batches in the following
        way:
         - At the beginning of every batch, a fixed fraction of the mandatory orders is chosen according to DGA costs
         - Then, all orders (mandatory and extra) used for DGA to complete the batch

        The numbers indicated by mandatory age completion should be chosen in accordance to the item completion goal in
        every phase and the number of newly added orders. The algorithm always picks orders to fulfill the completion
        goals and may exceed the minimum item goal already in doing so.
        """
        self.instance = instance
        self.order_ages = order_ages
        self.age_completion_list = mandatory_age_completion

    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        logger.info(
            f"Hard Anti Aging DGAPQ with completion targets: {self.age_completion_list}"
        )
        logger.info("Setting up problem instance.")
        remaining_item_goal = self.instance.parameters.min_number_requested_items
        max_batch_size = self.instance.parameters.max_orders_per_batch
        batches = []
        batch_orders = []
        batch_items = []
        warehouse_picklist_state = HardAntiAgingWarehousePicklistState(
            self.instance,
            self.order_ages,
            lambda age: (
                self.age_completion_list[age]
                if age < len(self.age_completion_list)
                else 0
            ),
        )

        logger.info("Starting batching.")

        item_goal_exceeded_warning_raised = False
        # Main loop
        while len(warehouse_picklist_state.mandatory_orders) > 0 or (
            remaining_item_goal > 0
            and len(warehouse_picklist_state.remaining_orders) > 0
        ):
            if remaining_item_goal < 0 and not item_goal_exceeded_warning_raised:
                logger.warning(
                    "The item goal has been reached, but more batches are required for the age completion targets."
                )
                item_goal_exceeded_warning_raised = True
            batch_time_1 = time.time()
            while len(batch_orders) < max_batch_size:
                next_order, next_items = (
                    warehouse_picklist_state.get_next_order_update_state()
                )
                if next_order is None or next_items is None:
                    break
                batch_orders.append(next_order)
                batch_items.extend(next_items)

            if cvrp_routing:
                picklists = cvrp_picklists(self.instance, batch_items)
            else:
                picklists = heuristic_picklists(
                    batch_items, self.instance.parameters.max_container_volume
                )

            batches.append(Batch(batch_orders, picklists))
            items_covered = len(batch_items)

            remaining_item_goal = remaining_item_goal - items_covered

            batch_time_2 = time.time()
            batch_duration = batch_time_2 - batch_time_1
            logger.info(
                f"Batch created with {len(batch_orders)} orders,{len(batches[-1].picklists)} picklists, "
                f"containing {items_covered} items. {remaining_item_goal} remaining items to cover. "
                f"Creation took {batch_duration} seconds."
            )

            assert len({bi.id for bi in batch_items}) == len(batch_items), (
                "Some item has been picked more than once"
            )

            batch_orders = []
            batch_items = []
            warehouse_picklist_state.reset_for_new_batch()
        logger.info("Batching completed.")

        return batches

    def backfill_ages(self):
        """
        Artificially age the current set of orders so that empty quotas for older orders are being filled.

        E.g., if 100 orders of age 10 are allowed, but only 20 are currently in the order set, pick 80 orders of age 9
        and increment their age to 10. If not enough orders are available from age 9, continue with age 8 etc. This is
        iteratively applied starting from the maximum allowed order age to move the current orders to the back as much
        as possible. Each order will not have their age decreased by this procedure.

        This is a helper routine to allow for artificial starts of the algorithm with a given set of orders without any
        real ages associated with them. Thereby, the following issue is fixed: if 5000 orders are added to the instance
        at the start of the phase, but only 1000 of age 0 are allowed to continue to the next phase, the algorithm must
        complete 4000 of those orders. This occurs at the beginning of the computation, where we populate the order set
        with more orders than are typically added from phase to phase. E.g., we could keep a set of 5000 orders at all
        times and add (and complete) 1000 in each phase. The 5000 initial orders start with the same age of 0 by
        default, which leads to different behaviour and an unwanted reduction of the order set.
        """
        max_age = len(self.age_completion_list)
        orders_by_age = [[] for _ in range(max_age)]
        for order, age in self.order_ages.items():
            orders_by_age[age].append(order)
        logger.info("Backfilling order ages")
        logger.debug(
            f"Age counts before backfilling: {[len(os) for os in orders_by_age]}"
        )
        for age in reversed(range(1, max_age)):
            n_to_fill = self.age_completion_list[age] - len(orders_by_age[age])
            for i in reversed(range(age)):
                if n_to_fill <= 0:
                    break
                orders_to_shift = random.sample(
                    orders_by_age[i], min(n_to_fill, len(orders_by_age[i]))
                )
                n_to_fill -= len(orders_to_shift)

                for order in orders_to_shift:
                    self.order_ages[order] = age
                    orders_by_age[i].remove(order)
                    orders_by_age[age].append(order)

        logger.debug(
            f"Age counts after backfilling: {[len(os) for os in orders_by_age]}"
        )


class HardAntiAgingDGAPQSolverLinear(HardAntiAgingDGAPQSolver):
    def __init__(self, instance: Instance, order_ages, max_age: int, n_new_orders: int):
        """
        Define the age completion targets according to a linear decline with 0 at the maximum age.
        """

        n_orders = len(instance.orders)

        def f(x):
            return max(0, math.ceil(n_new_orders * (max_age - x) / max_age))

        if sum(f(i) for i in range(max_age)) < n_orders:
            logger.warning(
                "The age completion slope is possibly too steep for the number of orders in the instance. Increase `max_age` accordingly."
            )

        age_completion_list = [f(i) for i in range(max_age)]

        super().__init__(instance, order_ages, age_completion_list)


class HardAntiAgingDGAPQSolverCutoff(HardAntiAgingDGAPQSolver):
    def __init__(self, instance: Instance, order_ages, max_age: int, n_new_orders: int):
        """
        Define the age completion targets according to a step function that only decreases from `n_new_orders` to 0 at `max_age`.
        """

        if n_new_orders * max_age < len(instance.orders):
            logger.warning(
                "The maximum completion age is too small for the number of orders in the instance. Increase `max_age` accordingly."
            )

        age_completion_list = [n_new_orders for i in range(max_age)]

        super().__init__(instance, order_ages, age_completion_list)
