"""This file defines very rudimentary solver approaches and wrappers for existing solver routines."""

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from batchingheuristics import Batch, Instance
from batchingheuristics.solvers.distance_greedy_algorithm.solver import greedy_solver
from batchingheuristics.solvers.picklist_computation import (
    cvrp_picklists,
    heuristic_picklists,
)


class PhaseSolver(ABC):
    def __init__(self, instance: Instance):
        self.instance = instance

    @abstractmethod
    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        pass


class RandomSolver(PhaseSolver):
    """
    Similar to the RDGA solver, we simply choose random orders. Here, we even randomly choose the item.
    The routings are optimized using a CVRP subroutine, unless `cvrp_routing` is set to false, in which case a heuristic
    is used.
    """

    def compute_batches(self, cvrp_routing=True) -> list[Batch]:
        remaining_item_goal = self.instance.parameters.min_number_requested_items
        remaining_orders = set(self.instance.orders.copy())
        items_by_article = defaultdict(list)

        for item in self.instance.warehouse_items:
            items_by_article[item.article].append(item)

        batches = []

        while remaining_item_goal > 0 and len(remaining_orders) > 0:
            batch_orders = []
            batch_items = []
            while len(batch_orders) < self.instance.parameters.max_orders_per_batch:
                new_order = random.sample(list(remaining_orders), 1)[0]
                batch_orders.append(new_order)
                remaining_orders.remove(new_order)
                remaining_item_goal -= len(new_order.positions)
                for article in new_order.positions:
                    batch_items.append(items_by_article[article].pop())
                if len(remaining_orders) == 0:
                    break
            if cvrp_routing:
                batches.append(
                    Batch(batch_orders, cvrp_picklists(self.instance, batch_items))
                )
            else:
                picklists = heuristic_picklists(
                    batch_items, self.instance.parameters.max_container_volume
                )
                batches.append(Batch(batch_orders, picklists))

        return batches


class ReferenceRDGASolver(PhaseSolver):
    """Uses the existing codebase to compute a solution by choosing random orders."""

    def compute_batches(self, cvrp_routing=False):
        return greedy_solver(
            self.instance, choose_random_order=True, cvrp_routing=cvrp_routing
        )


class ReferenceRDGASampleSolver(PhaseSolver):
    """
    Instead of using a single random order, we take the optimal out of a random selection of `sample_size` many orders.
    If the parameter is not given, use a sample size of `sqrt(num_orders)`.
    """

    def __init__(self, instance, sample_size: int | None = None):
        """
        If sample_size is None, use sqrt(n) many samples. Otherwise, a given int specifies the constant sample size.
        """
        self.instance = instance
        self.sample_size = sample_size

    def compute_batches(self, cvrp_routing=False):
        if self.sample_size is None:

            def choice_fun(x):
                return random.sample(list(x), round(math.sqrt(len(x))))

        else:
            k = self.sample_size

            def choice_fun(x):
                return random.sample(list(x), k)

        return greedy_solver(
            self.instance,
            choose_random_order=True,
            custom_random_choice=choice_fun,
            cvrp_routing=cvrp_routing,
        )


class ReferenceDGASolver(PhaseSolver):
    def __init__(self, instance: Instance):
        self.instance = instance

    def compute_batches(self, cvrp_routing=False) -> list[Batch]:
        return greedy_solver(self.instance, cvrp_routing=cvrp_routing)
