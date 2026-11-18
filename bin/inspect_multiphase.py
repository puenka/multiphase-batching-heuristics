import itertools
import json
import pathlib
from collections import Counter
import typer

import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import pandas as pd
import numpy as np

import batchingheuristics.inspection.check_solution_properties
from batchingheuristics.instances.definitions import Order, Instance, Parameters, Batch
from batchingheuristics.runners.multiphase_runner import ItemOrderChain
from loguru import logger
import gzip


import rich.table
import rich.console

console = rich.console.Console()


class RunnerStatistics:
    runner_phases: list
    new_orders_per_phase: list[list[Order]]

    order_age_added: dict
    order_age_removed: dict
    order_removed_count: list
    order_added_count: list

    total_stats: dict
    solver_parameters: dict

    def __init__(self):
        """Do not use this explicitly. Use helper init `from_file` or `from_runner` instead."""
        self.runner_phases = []
        self.new_orders_per_phase = []
        self.order_age_added = {}
        self.order_age_removed = {}
        self.order_removed_count = []
        self.order_added_count = []
        self.total_stats = {}
        self.solver_parameters = {}

        plt.set_loglevel("WARNING")

    @classmethod
    def from_runner(cls, runner):
        rs = cls()
        rs.runner_phases = runner.phases
        rs.new_orders_per_phase = runner.new_orders_per_phase
        rs.solver_parameters = runner.solver_params
        rs._initialize_order_stat_structures()
        return rs

    @classmethod
    def from_file(cls, fname: pathlib.Path, ioc: ItemOrderChain | None = None):
        rs = cls()

        logger.info(f"Loading phases from '{fname}'")
        with gzip.open(fname, mode="rt") as file:
            data = json.load(file)
            runner_phases_list = data["phases"]
            new_orders_per_phase_as_dict = data["new_orders_per_phase"]
            rs.solver_parameters = data.get("solver_parameters", {})
        ioc_path = fname.parent / "item-order-chain.json.gz"
        if ioc is None:
            logger.info(f"Loading instance from '{ioc_path}'")
            instance_ioc = ItemOrderChain.initialize_from_file(ioc_path)
        else:
            instance_ioc = ioc
        art_by_id = {a.id: a for a in instance_ioc.articles}
        order_by_id = {o.id: o for o in instance_ioc.order_list}
        item_by_id = {
            w.id: w for group in instance_ioc.item_grouped_list for w in group
        }
        item_by_id.update({w.id: w for w in instance_ioc.extra_items})
        for phase in runner_phases_list:
            new_instance = Instance()
            new_instance.articles = [art_by_id[a] for a in phase["articles"]]

            new_instance.batches = [
                Batch(
                    orders=[order_by_id[o] for o in b["orders"]],
                    picklists=[[item_by_id[w] for w in p] for p in b["picklists"]],
                )
                for b in phase["batches"]
            ]
            new_instance.orders = [order_by_id[o] for o in phase["orders"]]
            new_instance.warehouse_items = [
                item_by_id[w] for w in phase["warehouse_items"]
            ]
            new_instance.zones = phase["zones"]
            new_instance.parameters = Parameters(**phase["parameters"])
            if "stats" in phase.keys():
                new_instance.stats = phase["stats"]
            rs.runner_phases.append(new_instance)
        rs.new_orders_per_phase = [
            [order_by_id[od] for od in odl] for odl in new_orders_per_phase_as_dict
        ]
        rs._initialize_order_stat_structures()
        return rs

    # ------------- Common init helper function
    def _initialize_order_stat_structures(self):
        for i in range(len(self.new_orders_per_phase)):
            for new_order in self.new_orders_per_phase[i]:
                self.order_age_added[new_order.id] = i

        for i, phase in enumerate(self.runner_phases):
            orders_removed_ages = []
            for batch in phase.batches:
                for removed_order in batch.orders:
                    self.order_age_removed[removed_order.id] = i
                    orders_removed_ages.append(self.order_age_added[removed_order.id])
            self.order_removed_count.append(Counter(orders_removed_ages))

        self.order_added_count = [len(p) for p in self.new_orders_per_phase]

        # Compute total stats
        self.total_stats = {}
        self.total_stats["nbr_rp"] = (
            len(self.runner_phases) if len(self.runner_phases) > 0 else 1
        )
        self.total_stats["total_cost"] = sum(
            [
                p.stats["objective_value"]
                for p in self.runner_phases
                if hasattr(p, "stats")
            ]
        )
        self.total_stats["all_feasible"] = all(
            [p.stats["feasible"] for p in self.runner_phases if hasattr(p, "stats")]
        )
        self.total_stats["nbr_picklists"] = sum(
            [len(b.picklists) for p in self.runner_phases for b in p.batches]
        )
        self.total_stats["nbr_orders"] = sum(
            [len(b.orders) for p in self.runner_phases for b in p.batches]
        )
        self.total_stats["nbr_items"] = sum(
            [
                len(o.positions)
                for p in self.runner_phases
                for b in p.batches
                for o in b.orders
            ]
        )

    # ------------- Stats for given phases
    def print_order_age_statistics(self, phases):
        for phase in phases:
            picked_orders = list(
                itertools.chain(*[b.orders for b in self.runner_phases[phase].batches])
            )
            all_orders = self.runner_phases[phase].orders
            picked_avg_age = 0
            for order in picked_orders:
                picked_avg_age += phase - self.order_age_added.get(
                    order.id, 100_000_000
                )
            picked_avg_age = (
                picked_avg_age / len(picked_orders) if len(picked_orders) > 0 else -1
            )

            total_avg_age = sum(
                [
                    phase - self.order_age_added.get(o.id, 100_000_000)
                    for o in all_orders
                ]
            )
            total_avg_age = (
                total_avg_age / len(all_orders) if len(all_orders) > 0 else -1
            )

            order_birth_dist = Counter([self.order_age_added[o.id] for o in all_orders])
            order_picked_birth_dist = Counter(
                [self.order_age_added[o.id] for o in picked_orders]
            )

            avg_length_picked = (
                sum([len(o.positions) for o in picked_orders]) / len(picked_orders)
                if len(picked_orders) > 0
                else 0
            )
            avg_length_total = (
                sum([len(o.positions) for o in all_orders]) / len(all_orders)
                if len(all_orders) > 0
                else 0
            )

            table = rich.table.Table(
                "Statistic", "Value", title=f"Phase {phase} general statistics"
            )

            table.add_row("Nbr of picked orders", f"{len(picked_orders)}")
            table.add_row("Avg picked order age", f"{picked_avg_age:.2f}")
            table.add_row("Avg total order age", f"{total_avg_age:.2f}")

            table.add_row("Avg #positions in picked orders", f"{avg_length_picked:.2f}")
            table.add_row(
                "Avg #positions in warehouse orders", f"{avg_length_total:.2f}"
            )

            table.add_row("Total birth distribution", f"{order_birth_dist}")
            table.add_row("Picked birth distribution", f"{order_picked_birth_dist}")

            console.print(table)

    def print_order_fulfillment_rate(self, phases):
        for phase in phases:
            count = self.order_added_count[phase]
            console.print(
                f"Phase {phase} - Total count of added orders from phase: {count}"
            )
            if count > 0:
                console.print("Fulfilled by phase:", end="")
                total_removed = 0
                for next_phase in range(phase, len(self.runner_phases)):
                    total_removed += self.order_removed_count[next_phase][phase]
                    console.print(
                        f"\t {next_phase}: {total_removed} ({round(total_removed / count * 100)}%)",
                    )
                print("")

    def print_item_dist_statistics(self, phases):
        def origin_dist(item):
            return abs(item.aisle) + abs(item.row)

        for phase in phases:
            inst = self.runner_phases[phase]
            warehouse_items = inst.warehouse_items
            picked_orders = list(itertools.chain(*[b.orders for b in inst.batches]))
            all_orders = inst.orders

            avg_w_distance = (
                (
                    sum([origin_dist(item) for item in warehouse_items])
                    / len(warehouse_items)
                )
                if len(warehouse_items) > 0
                else 0
            )

            central_item_per_article = {}
            for item in warehouse_items:
                if item.article not in central_item_per_article.keys():
                    central_item_per_article[item.article] = item
                elif origin_dist(central_item_per_article[item.article]) > origin_dist(
                    item
                ):
                    central_item_per_article[item.article] = item

            avg_po_distance = 0
            for order in picked_orders:
                avg_po_distance += sum(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_po_distance = (
                avg_po_distance / len(picked_orders) if len(picked_orders) > 0 else 0
            )

            avg_o_distance = 0
            for order in all_orders:
                avg_o_distance += sum(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_o_distance = (
                avg_o_distance / len(all_orders) if len(all_orders) > 0 else 0
            )

            remaining_orders = set(all_orders) - set(picked_orders)
            avg_r_distance = 0
            for order in remaining_orders:
                avg_r_distance += sum(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_r_distance = (
                avg_r_distance / len(remaining_orders)
                if len(remaining_orders) > 0
                else 0
            )

            newly_added_orders = self.new_orders_per_phase[phase]
            avg_n_distance = 0
            for order in newly_added_orders:
                # avg_n_distance += sum([origin_dist(central_item_per_article[art]) for art in order.positions])
                pass
            avg_n_distance = (
                avg_n_distance / len(newly_added_orders)
                if len(newly_added_orders) > 0
                else 0
            )

            table = rich.table.Table(
                "Avg. distance of all ... (most central item)",
                "Value",
                title=f"Phase {phase} distance statistics",
            )

            console.print(
                f"Phase {phase} - Total count of orders: {len(all_orders)}, "
                f"#picked orders: {len(picked_orders)}, "
                f"# warehouse items {len(warehouse_items)}"
            )

            table.add_row("items", f"{avg_w_distance:.2f}")
            table.add_row("orders", f"{avg_o_distance:.2f}")
            table.add_row("picked orders", f"{avg_po_distance:.2f}")
            table.add_row("remaining orders", f"{avg_r_distance:.2f}")

            console.print(table)

    def print_order_age_tail_stats(self, phases):
        for phase in phases:
            order_birth_dist = Counter({i: 0 for i in range(phase)})
            order_birth_dist.update(
                [self.order_age_added[o.id] for o in self.runner_phases[phase].orders]
            )
            order_count = len(self.runner_phases[phase].orders)
            age_data = pd.DataFrame(
                [
                    {"birth": int(age), "count": count}
                    for age, count in order_birth_dist.items()
                ]
            )
            age_data_cum = age_data.cumsum()
            console.print(f"Phase {phase}", end="")
            for offset in [20, 15, 10, 5]:
                if phase > offset:
                    console.print(
                        f"\tT-{offset} older orders: {round(age_data_cum['count'].get(phase - offset, 0) / order_count * 100)}%",
                        end="",
                    )
            print()

    def print_order_eval_stats(self, phases):
        for phase in phases:
            console.print(f"Phase {phase}", end="")
            stats = self.runner_phases[phase].stats
            console.print(
                f"\t cost: {stats['objective_value']}\t feasible: {stats['feasible']}\t #picklists: {stats['nbr_picklists']}"
            )

    # Stats for the full run
    def print_total_cost(self):
        console.print(
            f"Total cost of {self.total_stats['total_cost']} over {self.total_stats['nbr_rp']} phases with average cost/phase {self.total_stats['total_cost'] / self.total_stats['nbr_rp']:.2f}."
            f" Average #picklists: {self.total_stats['nbr_picklists'] / self.total_stats['nbr_rp']:.2f}.\nTotal number of Orders: {self.total_stats['nbr_orders']} for a cost of "
            f"{self.total_stats['total_cost'] / self.total_stats['nbr_orders']:.2f} per order. Total number of items: {self.total_stats['nbr_items']} for a cost of "
            f"{self.total_stats['total_cost'] / self.total_stats['nbr_items']:.2f} per item. {(self.total_stats['nbr_items'] / self.total_stats['nbr_picklists']):.2f} items/picklist."
            f" {(self.total_stats['total_cost'] / self.total_stats['nbr_picklists']):.2f} cost/picklist. \nAll feasible: {self.total_stats['all_feasible']}"
        )

    # Visualization methods
    def plot_cost_stats(self):
        cost_data = []
        picklist_data = []
        for p_ind in range(len(self.runner_phases)):
            inst = self.runner_phases[p_ind]
            cost_data.append((p_ind, inst.stats["objective_value"]))
            picklist_data.append((p_ind, sum([len(b.picklists) for b in inst.batches])))

        cost_cumsum = []
        cumsum = 0
        for ind, cost in cost_data:
            cumsum += cost
            cost_cumsum.append((ind, cumsum))

        print(cost_cumsum)
        print(cost_data)
        # Initializing axes and so on (inspired by: https://stackoverflow.com/questions/7733693/)
        fig, ax = plt.subplots()
        axes = [ax, ax.twinx(), ax.twinx()]
        fig.subplots_adjust(right=0.75)
        axes[-1].spines["right"].set_position(("axes", 1.2))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)

        # Plotting of different data values (each on their own axis)
        axes[0].plot(cost_data, color="Red")
        axes[0].set_ylabel("Cost of phase", color="Red")
        axes[1].plot(cost_cumsum, color="Purple")
        axes[1].set_ylabel("Cumulative cost of phases", color="Purple")
        axes[2].plot(
            [x[0] for x in picklist_data], [x[1] for x in picklist_data], color="Blue"
        )
        axes[2].set_ylabel("Number of picklists in phase", color="Blue")
        plt.show()

    def plot_order_fulfillment_statistics(self):
        age_array = np.zeros((len(self.runner_phases), len(self.runner_phases)))

        for phase in range(len(self.runner_phases)):
            order_birth_dist = Counter(
                [self.order_age_added[o.id] for o in self.runner_phases[phase].orders]
            )
            for age, count in order_birth_dist.items():
                age_array[age, phase] = count
        plt.stackplot(
            np.arange(len(self.runner_phases)),
            age_array,
            labels=np.arange(len(self.runner_phases)),
        )
        plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.title("Order fulfillment by birth phase")
        plt.xlabel("phase")
        plt.ylabel("order count")
        plt.legend(title="birth")
        plt.show()

    def plot_order_birth_statistics(self, phase_index):
        order_birth_dist = Counter({i: 0 for i in range(phase_index)})
        order_birth_dist.update(
            [self.order_age_added[o.id] for o in self.runner_phases[phase_index].orders]
        )
        order_birth_picked = Counter({i: 0 for i in range(phase_index)})
        order_birth_picked.update(
            [
                self.order_age_added[o.id]
                for b in self.runner_phases[phase_index].batches
                for o in b.orders
            ]
        )

        age_data = pd.DataFrame(
            [
                {"birth": int(age), "count": count, "picked": order_birth_picked[age]}
                for age, count in order_birth_dist.items()
            ]
        )
        sns.barplot(age_data, x="birth", y="count")
        sns.barplot(age_data, x="birth", y="picked")
        plt.twinx()
        plt.plot(age_data["birth"], age_data["count"].cumsum(), c="red")
        plt.ylabel("cumulative sum")
        plt.suptitle(
            f"Phase: {phase_index}. Total #Orders: {len(self.runner_phases[phase_index].orders)}"
        )
        plt.show()

    def plot_order_dist_statistics(self, averaged_by_order=False):
        phase_count = len(self.runner_phases)
        dist_data = pd.DataFrame()

        def origin_dist(item):
            return abs(item.aisle) + abs(item.row)

        for phase in range(phase_count):
            inst = self.runner_phases[phase]
            warehouse_items = inst.warehouse_items
            picked_orders = list(itertools.chain(*[b.orders for b in inst.batches]))
            all_orders = inst.orders

            if averaged_by_order:

                def f(x):  # pyright: ignore[reportRedeclaration]
                    return sum(x) / len(x) if len(x) > 0 else 0
            else:

                def f(x):
                    return sum(x)

            avg_w_distance = (
                (
                    sum([origin_dist(item) for item in warehouse_items])
                    / len(warehouse_items)
                )
                if len(warehouse_items) > 0
                else 0
            )

            central_item_per_article = {}
            for item in warehouse_items:
                if item.article not in central_item_per_article.keys():
                    central_item_per_article[item.article] = item
                elif origin_dist(central_item_per_article[item.article]) > origin_dist(
                    item
                ):
                    central_item_per_article[item.article] = item

            avg_po_distance = 0
            for order in picked_orders:
                avg_po_distance += f(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_po_distance = (
                avg_po_distance / len(picked_orders) if len(picked_orders) > 0 else 0
            )

            avg_o_distance = 0
            for order in all_orders:
                avg_o_distance += f(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_o_distance = (
                avg_o_distance / len(all_orders) if len(all_orders) > 0 else 0
            )

            remaining_orders = set(all_orders) - set(picked_orders)
            avg_r_distance = 0
            for order in remaining_orders:
                avg_r_distance += f(
                    [
                        origin_dist(central_item_per_article[art])
                        for art in order.positions
                    ]
                )
            avg_r_distance = (
                avg_r_distance / len(remaining_orders)
                if len(remaining_orders) > 0
                else 0
            )

            dist_data = pd.concat(
                [
                    dist_data,
                    pd.DataFrame(
                        [
                            {
                                "phase": phase,
                                "dist": avg_w_distance,
                                "type": "All items",
                            },
                            {
                                "phase": phase,
                                "dist": avg_po_distance,
                                "type": "Picked orders",
                            },
                            {
                                "phase": phase,
                                "dist": avg_o_distance,
                                "type": "All orders",
                            },
                            {
                                "phase": phase,
                                "dist": avg_r_distance,
                                "type": "Remaining orders",
                            },
                        ]
                    ),
                ]
            )
        sns.lineplot(dist_data, x="phase", y="dist", hue="type")
        plt.show()

    # ------------- Function interfaces to methods defined in check_solution_properties.py

    def show_item_heatmap(self, phase_index):
        fig = batchingheuristics.inspection.check_solution_properties.show_item_heatmap(
            self.runner_phases[phase_index]
        )
        fig.suptitle(f"Phase {phase_index}")
        plt.show()

    def show_distance_distribution(self, phase_index):
        fig = batchingheuristics.inspection.check_solution_properties.show_distance_distributions(
            self.runner_phases[phase_index]
        )
        fig.suptitle(f"Phase {phase_index}")
        plt.show()

    def test_order_distribution(self, phase_index):
        all_orders = list(itertools.chain(*self.new_orders_per_phase[0:phase_index]))
        completed_orders = list(
            itertools.chain(*[p.batches for p in self.runner_phases[0:phase_index]])
        )
        batchingheuristics.inspection.check_solution_properties.test_order_distributions(
            all_orders, completed_orders
        )


def main(
    results_file: pathlib.Path,
    phases: list[int] = [],
    all_plots: bool = False,
    plot_by_birth: bool = False,
    plot_dist: bool = False,
    plot_costs: bool = False,
    plot_heatmap: bool = False,
):
    rs = RunnerStatistics.from_file(results_file)

    console.print(
        f"New orders per phase: {[len(orders) for orders in rs.new_orders_per_phase]}."
    )

    rs.print_order_age_statistics(phases)
    rs.print_order_fulfillment_rate(phases)
    rs.print_item_dist_statistics(phases)
    rs.print_order_age_tail_stats(phases)

    rs.print_order_eval_stats(phases)
    rs.print_total_cost()

    if all_plots or plot_by_birth:
        rs.plot_order_fulfillment_statistics()
        for p in phases:
            rs.plot_order_birth_statistics(p)

    if all_plots or plot_dist:
        rs.plot_order_dist_statistics()
        rs.plot_order_dist_statistics(True)

    if all_plots or plot_costs:
        rs.plot_cost_stats()

    if all_plots or plot_heatmap:
        for p in phases:
            rs.show_item_heatmap(p)


if __name__ == "__main__":
    typer.run(main)
