import gzip
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from batchingheuristics.instances.definitions import (
    Article,
    Batch,
    Instance,
    Order,
    Parameters,
    WarehouseItem,
)
from batchingheuristics.instances.multiphase_generator import ItemOrderChain
from batchingheuristics.solvers import (
    HardAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolverCutoff,
    HardAntiAgingDGAPQSolverLinear,
    SoftAntiAgingDGAPQSolver,
)
from batchingheuristics.solvers.simple_solvers import PhaseSolver

SOLVERS_WITH_ARGS = (
    SoftAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolverLinear,
    HardAntiAgingDGAPQSolverCutoff,
)
HARD_DGAPQ_SOLVERS = (
    HardAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolverLinear,
    HardAntiAgingDGAPQSolverCutoff,
)


class PhaseEncoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case WarehouseItem() | Order() | Article():
                return o.id
            case Batch() | Instance() | Parameters():
                return o.__dict__
            case _:
                return super().default(o)


class MultiphaseRunner:
    item_order_chain: ItemOrderChain
    solver_type: type[
        PhaseSolver
        | SoftAntiAgingDGAPQSolver
        | HardAntiAgingDGAPQSolver
        | HardAntiAgingDGAPQSolverLinear
        | HardAntiAgingDGAPQSolverCutoff
    ]
    warehouse_items: list[WarehouseItem]
    item_ages: dict[WarehouseItem, int]
    order_ages: dict[Order, int]
    orders: list[Order]
    phases: list[Instance]
    instance_params: Parameters

    new_orders_per_phase: list[list[Order]]

    def __init__(
        self,
        solver_type: type[
            PhaseSolver
            | SoftAntiAgingDGAPQSolver
            | HardAntiAgingDGAPQSolver
            | HardAntiAgingDGAPQSolverLinear
            | HardAntiAgingDGAPQSolverCutoff
        ],
        instance_dir: Path,
        **solver_params,
    ):
        if (
            instance_dir / "item-order-chain.json.gz"
        ).exists():  # try the compressed file first
            self.item_order_chain = ItemOrderChain.initialize_from_file(
                instance_dir / "item-order-chain.json.gz"
            )
        else:
            self.item_order_chain = ItemOrderChain.initialize_from_file(
                instance_dir / "item-order-chain.json"
            )
        self.solver_type = solver_type
        self.warehouse_items = []
        self.item_ages = {}
        self.order_ages = {}
        self.orders = []
        self.phases = []
        self.instance_params = Parameters(
            **json.loads((instance_dir / "parameters.json").read_text())
        )
        self.solver_params = solver_params
        self.new_orders_per_phase = []

    def run_init(self, num_orders: int, num_items: int, cvrp_routing=True):
        """
        Run the first phase. In this case, the number of items and orders can be chosen arbitrarily in order to
        establish an arbitrary amount of additional items (excluding the ones given in the orders) that are also
        available in the warehouse.
        `num_items` is the total number of items, i.e. the sum of ordered items and excess items in the warehouse.

        Giving a fixed seed allows for reproducibility in the sequence of additional items.
        """
        logger.info("Running initial phase")
        self.warehouse_items, self.orders = self.item_order_chain.initial_order_request(
            num_items, num_orders
        )
        logger.info(
            f"Starting warehouse with {len(self.warehouse_items)} items for {len(self.orders)} orders."
        )
        self.item_ages = {item: 0 for item in self.warehouse_items}
        self.order_ages = {order: 0 for order in self.orders}
        self.new_orders_per_phase.append(self.orders.copy())

        phase_instance = self._compute_phase_batches(
            cvrp_routing, dgapq_backfill=self.solver_type in HARD_DGAPQ_SOLVERS
        )
        phase_instance.evaluate()
        self.phases.append(phase_instance)
        self._age()

        logger.info(
            f"The phase completed {get_nr_picked_items(phase_instance)} items "
            f"for {sum(len(batch.orders) for batch in phase_instance.batches)} orders"
        )

    def run_phase(self, num_items, cvrp_routing=True):
        """
        Run a phase without adding any excess items to the warehouse. If this is run without a call to
        `MultiphaseRunner.run_init` it corresponds to the warehouse being filled precisely with the ordered items in the
        first phase.
        """
        logger.info("Running next phase")

        new_items, new_orders = self.item_order_chain.return_items_and_orders(num_items)

        for item in new_items:
            self.item_ages[item] = 0
        for order in new_orders:
            self.order_ages[order] = 0
        self.new_orders_per_phase.append(new_orders)
        self.warehouse_items.extend(new_items)
        self.orders.extend(new_orders)
        logger.info(
            f"Adding {len(new_items)} items for {len(new_orders)} orders to the warehouse. Total now: {len(self.orders)} orders."
        )

        phase_instance = self._compute_phase_batches(cvrp_routing)

        phase_instance.evaluate()
        self.phases.append(phase_instance)
        self._age()

        logger.info(
            f"The phase completed {get_nr_picked_items(phase_instance)} items "
            f"for {sum(len(batch.orders) for batch in phase_instance.batches)} orders"
        )

    def _compute_phase_batches(self, cvrp_routing, dgapq_backfill=False) -> Instance:
        """Creates a single phase instance to be used with the BatchSolver"""
        instance = Instance()
        instance.articles = [item.article for item in self.warehouse_items]
        instance.orders = self.orders.copy()
        instance.warehouse_items = self.warehouse_items.copy()
        instance.zones = self.item_order_chain.zones
        instance.parameters = self.instance_params
        if self.solver_type in SOLVERS_WITH_ARGS:
            if self.solver_type in HARD_DGAPQ_SOLVERS:
                solver = self.solver_type(
                    instance, self.order_ages, **self.solver_params
                )
                if dgapq_backfill:
                    solver.backfill_ages()
                instance.batches = solver.compute_batches(cvrp_routing)
            else:
                instance.batches = self.solver_type(
                    instance, self.order_ages, **self.solver_params
                ).compute_batches(cvrp_routing)
        elif issubclass(self.solver_type, PhaseSolver):
            instance.batches = self.solver_type(instance).compute_batches(cvrp_routing)

        batched_items = set(
            itertools.chain(*[batch.get_all_items() for batch in instance.batches])
        )
        batched_orders = set(
            itertools.chain(*[batch.orders for batch in instance.batches])
        )
        self.warehouse_items = [
            item for item in self.warehouse_items if item not in batched_items
        ]
        self.orders = [order for order in self.orders if order not in batched_orders]

        return instance

    def _age(self):
        for item in self.warehouse_items:
            self.item_ages[item] += 1
        for order in self.orders:
            self.order_ages[order] += 1

    def get_agecount_warehouse(self) -> Counter:
        ages = []
        for item in self.warehouse_items:
            ages.append(self.item_ages[item])
        return Counter(ages)

    def get_agecount_picked(self) -> Counter:
        ages = []
        for item, age in self.item_ages.items():
            if item not in self.warehouse_items:
                ages.append(age)
        return Counter(ages)

    def get_agecount_orders_completed(self) -> Counter:
        count = Counter()
        for phase in self.phases:
            for batch in phase.batches:
                for order in batch.orders:
                    count[self.order_ages[order]] += 1
        return count

    def get_agecount_orders_open(self) -> Counter:
        return Counter([self.order_ages[order] for order in self.orders])

    def get_picklist_costs(self):
        """
        Returns a nested list of costs computed for each picklist.
        The list is nested by phase/batch/picklist
        """
        return [
            [
                [phase.picklist_cost(picklist) for picklist in batch.picklists]
                for batch in phase.batches
            ]
            for phase in self.phases
        ]

    def get_batch_costs(self):
        return [[sum(batch) for batch in phase] for phase in self.get_picklist_costs()]

    def get_phase_costs(self):
        return [sum(batches) for batches in self.get_batch_costs()]

    def get_total_cost(self):
        return sum(self.get_phase_costs())

    def get_warehouse_distances(self):
        instance = Instance()
        instance.warehouse_items = self.warehouse_items.copy()
        instance.zones = self.item_order_chain.zones
        instance.parameters = self.instance_params
        return {
            item: instance.distance_to_conveyor(item)
            for item in instance.warehouse_items
        }

    def get_order_distances(self):
        """
        Computes an estimate for the (average) cost of each order.
        For each article in the order, the estimate adds the distance of the corresponding item closest to the conveyor,
        while disregarding potentially taking the same item twice. The total distance is then divided by the number of
        positions in the order.
        """

        instance = Instance()
        instance.warehouse_items = self.warehouse_items.copy()
        instance.zones = self.item_order_chain.zones
        instance.parameters = self.instance_params

        items_per_article = defaultdict(list)

        for item in instance.warehouse_items:
            items_per_article[item.article].append(item)

        article_dists = {}
        order_dists = {}

        for order in self.orders:
            cost = 0
            for article in order.positions:
                if article not in article_dists:
                    article_dists[article] = min(
                        instance.distance_to_conveyor(item)
                        for item in items_per_article[article]
                    )
                cost += article_dists[article]
            order_dists[order] = cost / len(order.positions)

        return order_dists

    def write(self, fname, params=None):
        """Store the computed phases to the file `fname` as a gzip compressed json in text mode."""
        obj_to_write = {
            "phases": self.phases,
            "new_orders_per_phase": self.new_orders_per_phase,
            "solver_parameters": params,
        }
        target_path = Path(fname)
        logger.info(f"Store computed phases to '{target_path}'.")

        with gzip.open(target_path, mode="wt", encoding="utf-8") as f:
            json.dump(obj_to_write, f, indent=4, cls=PhaseEncoder)

    def __repr__(self):
        repr_str = (
            f"MultiphaseRunner with solver: {self.solver_type}\n"
            f"Inventory: {len(self.warehouse_items)} items for {len(self.orders)} orders\n"
            f"In total {len(self.phases)} phases have been computed for a total cost of "
            f"{sum(map(get_instance_cost, self.phases))}"
        )
        return repr_str


def get_nr_picked_items(instance: Instance):
    return sum(
        len(picklist) for batch in instance.batches for picklist in batch.picklists
    )


def get_instance_cost(instance: Instance):
    return sum(
        instance.picklist_cost(picklist)
        for batch in instance.batches
        for picklist in batch.picklists
    )
