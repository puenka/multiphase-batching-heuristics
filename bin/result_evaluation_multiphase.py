# %%
from typing import Any, Optional
from inspect_multiphase import RunnerStatistics
import pathlib
import polars as pl
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d  # noqa: F401
import numpy as np
import itertools
from batchingheuristics import ItemOrderChain, Instance, Order
from loguru import logger
from concurrent.futures import ProcessPoolExecutor


algo_name_map = {
    "dgapq": "\\fdga",
    "dgapq-cutoff": "\\fdgacutoffn",
    "dgapq-linear": "\\fdgalinearn",
    "dgapq-soft": "\\fdgasoftg",
    "neighbour-aisle": "\\ldga",
    "rdga-sqrt": "\\rdgasqrt",
}


def compute_sorted_dists(
    instance: Instance, orders: Optional[list[Order]] = None
) -> float:
    if orders is None:
        orders = instance.orders
    a_i_dict = instance.get_article_item_dict()
    article_counts = Counter(itertools.chain(*[o.positions for o in instance.orders]))
    sorted_dists = dict()
    for article in article_counts.keys():
        sorted_dists[article] = sorted(
            instance.distance_to_conveyor(item) for item in a_i_dict[article]
        )

    return sum(sum(sorted_dists[a][:m_a]) for a, m_a in article_counts.items()) / sum(
        len(o.positions) for o in instance.orders
    )


def compute_min_dists(
    instance: Instance, orders: Optional[list[Order]] = None
) -> float:
    if orders is None:
        orders = instance.orders
    a_i_dict = instance.get_article_item_dict()
    ordered_articles = list(itertools.chain(*[o.positions for o in orders]))
    min_dists = dict()
    for article in set(ordered_articles):
        min_dists[article] = min(
            instance.distance_to_conveyor(item) for item in a_i_dict[article]
        )

    return sum(min_dists[a] for a in ordered_articles) / len(ordered_articles)


def load_instance(fname: pathlib.Path, ioc: ItemOrderChain):
    rs = RunnerStatistics.from_file(fname, ioc)

    rs_data: dict[str, Any] = {"file": fname.name}
    rs_data.update(rs.total_stats)
    rs_data["solver_parameters"] = rs.solver_parameters
    rs_data["avg_order_length_init"] = np.mean(
        [len(o.positions) for o in rs.runner_phases[0].orders]
    )
    rs_data["avg_order_length_final"] = np.mean(
        [len(o.positions) for o in rs.runner_phases[-1].orders]
    )
    rs_data["avg_sorted_distance_init"] = compute_sorted_dists(rs.runner_phases[0])
    rs_data["avg_sorted_distance_final"] = compute_sorted_dists(rs.runner_phases[-1])
    rs_data["max_order_age_final"] = 100 - min(
        [rs.order_age_added[o.id] for o in rs.runner_phases[-1].orders]
    )
    rs_data["avg_order_age_final"] = 100 - np.mean(
        [rs.order_age_added[o.id] for o in rs.runner_phases[-1].orders]
    )
    rs_data["median_order_age_final"] = 100 - np.median(
        [rs.order_age_added[o.id] for o in rs.runner_phases[-1].orders]
    )

    # Add min distance data
    rs_data["avg_min_distance_init_all"] = compute_min_dists(rs.runner_phases[0])
    batched_orders_init = list(
        itertools.chain(*[b.orders for b in rs.runner_phases[0].batches])
    )
    rs_data["avg_min_distance_init_batched"] = compute_min_dists(
        rs.runner_phases[0], batched_orders_init
    )
    rs_data["avg_min_distance_init_unbatched"] = compute_min_dists(
        rs.runner_phases[0],
        [o for o in rs.runner_phases[0].orders if o not in batched_orders_init],
    )

    rs_data["avg_min_distance_final_all"] = compute_min_dists(rs.runner_phases[-1])
    batched_orders_final = list(
        itertools.chain(*[b.orders for b in rs.runner_phases[-1].batches])
    )
    rs_data["avg_min_distance_final_batched"] = compute_min_dists(
        rs.runner_phases[-1], batched_orders_final
    )
    rs_data["avg_min_distance_final_unbatched"] = compute_min_dists(
        rs.runner_phases[-1],
        [o for o in rs.runner_phases[-1].orders if o not in batched_orders_final],
    )

    return rs_data


def data_prep(data):
    df = pl.DataFrame(data).with_columns(
        cost_per_item=pl.col("total_cost") / pl.col("nbr_items"),
        cost_per_order=pl.col("total_cost") / pl.col("nbr_orders"),
    )
    df = df.unnest("solver_parameters").with_columns(
        algorithm=pl.col("file").str.extract(r"^([\w-]+)-100")
    )

    avg_df = (
        df.group_by("algorithm", "max_age", "n_new_orders", "gamma")
        .agg(
            avg_total_cost=pl.col("total_cost").mean().round(0),
            avg_cost_per_item=pl.col("cost_per_item").mean().round(2),
            avg_cost_per_order=pl.col("cost_per_order").mean().round(2),
            avg_sorted_distance_init=pl.col("avg_sorted_distance_init").mean().round(2),
            avg_sorted_distance_final=pl.col("avg_sorted_distance_final")
            .mean()
            .round(2),
            avg_order_length_init=pl.col("avg_order_length_init").mean().round(2),
            avg_order_length_final=pl.col("avg_order_length_final").mean().round(2),
            max_order_age_final=pl.col("max_order_age_final").mean().round(2),
            avg_order_age_final=pl.col("avg_order_age_final").mean().round(2),
            avg_min_distance_init_all=pl.col("avg_min_distance_init_all")
            .mean()
            .round(2),
            avg_min_distance_init_batched=pl.col("avg_min_distance_init_batched")
            .mean()
            .round(2),
            avg_min_distance_init_unbatched=pl.col("avg_min_distance_init_unbatched")
            .mean()
            .round(2),
            avg_min_distance_final_all=pl.col("avg_min_distance_final_all")
            .mean()
            .round(2),
            avg_min_distance_final_batched=pl.col("avg_min_distance_final_batched")
            .mean()
            .round(2),
            avg_min_distance_final_unbatched=pl.col("avg_min_distance_final_unbatched")
            .mean()
            .round(2),
            median_order_age_final=pl.col("median_order_age_final").mean().round(2),
        )
        .sort("algorithm", "max_age", "gamma")
    )
    return avg_df


def multiphase_performance(df):
    for row in df.with_columns(
        rel_perf=pl.col("avg_total_cost")
        / pl.col("avg_total_cost").filter(pl.col("algorithm") == "dgapq")
    ).iter_rows(named=True):
        print(
            f"{algo_name_map[row['algorithm']]}{'{' + str(row['max_age']) + '}' if row['max_age'] else ''}{'{' + str(row['gamma']) + '}' if row['gamma'] else ''} & "
            f"{row['avg_total_cost']:.0f} & "
            f"${(row['rel_perf'] - 1) * 100:+.1f}\\% $ & "
            f"{row['avg_cost_per_item']} & "
            f"{row['avg_cost_per_order']} \\\\"
        )


def multiphase_age_stats(df):
    # Evaluate the cost in terms of different metrics from the computed multiphase data.
    for row in df.iter_rows(named=True):
        print(
            f"{algo_name_map[row['algorithm']]}{'{' + str(row['max_age']) + '}' if row['max_age'] else ''}{'{' + str(row['gamma']) + '}' if row['gamma'] else ''} & "
            f"{row['avg_order_length_final']} & "
            f"{row['max_order_age_final']} & "
            f"{row['avg_order_age_final']} & "
            f"{row['median_order_age_final']} \\\\"
        )


def plot_age_stats(fname, title=None):
    fname = pathlib.Path(fname)
    if title is None:
        title = fname.stem
    rs = RunnerStatistics.from_file(fname)
    print(rs.solver_parameters)

    age_counters = [
        Counter([i - rs.order_age_added[o.id] for o in rs.runner_phases[i].orders])
        for i in range(0, len(rs.runner_phases))
    ]
    age_list = np.array(
        [[age_counter.get(i, 0) for i in range(100)] for age_counter in age_counters]
    ).transpose()

    plt.figure()
    plt.title(title)
    plt.pcolor(
        age_list, norm=colors.LogNorm(vmin=1, vmax=age_list.max()), cmap="cubehelix_r"
    )
    plt.colorbar(label="Number of orders of each age")
    plt.xlabel("Phase")
    plt.ylabel("Order age")
    plt.savefig(
        pathlib.Path("plots") / (title.replace(" ", "_") + "_agestats.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_age_stats_3d(fname):
    rs = RunnerStatistics.from_file(fname)
    print(rs.solver_parameters)

    age_counters = [
        Counter([i - rs.order_age_added[o.id] for o in rs.runner_phases[i].orders])
        for i in range(0, len(rs.runner_phases))
    ]
    age_list = np.array(
        [[age_counter.get(i, 0) for i in range(100)] for age_counter in age_counters]
    ).transpose()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X = list(range(100))
    Y = list(range(101))

    XX, YY = np.meshgrid(X, Y)
    ax.plot_surface(XX, YY, age_list, cmap="afmhot_r")  # pyright: ignore[reportAttributeAccessIssue]


def compute_tables(instance_size: str, instance_type: str, n_concurrent_jobs: int = 4):
    base_path = pathlib.Path("data/instances-multiphase")
    data = []
    for instance_dir in list(base_path.glob(f"{instance_size}{instance_type}-?/")):
        logger.info(f"Instance directory {instance_dir}")
        if n_concurrent_jobs > 0:
            with ProcessPoolExecutor(max_workers=n_concurrent_jobs) as ex:
                ioc = ItemOrderChain.initialize_from_file(
                    instance_dir / "item-order-chain.json.gz"
                )
                futures = []
                for fname in list(instance_dir.glob("*100phases*.json.gz")):
                    futures.append(ex.submit(load_instance, fname, ioc))
                data.extend(f.result() for f in futures)
        else:
            ioc = ItemOrderChain.initialize_from_file(
                instance_dir / "item-order-chain.json.gz"
            )
            for fname in list(instance_dir.glob("*100phases*.json.gz")):
                data.append(load_instance(fname, ioc))

    df = data_prep(data)
    df.write_csv(base_path / f"multiphase_stats_{instance_size}{instance_type}.csv")

    print(f"Multiphase performance {instance_size}{instance_type}")
    multiphase_performance(df)
    print(f"Multiphase age stats {instance_size}{instance_type}")

    multiphase_age_stats(df)


def plot_examples(instance_size: str, instance_type: str):
    base_dir = f"data/instances-multiphase/{instance_size}{instance_type}-0"

    for file, title in [
        ("dgapq-100phases-result.json.gz", f"FDGA {instance_size}{instance_type}"),
        (
            "dgapq-soft-100phases-result_f5eaae6.json.gz",
            f"FDGA-Soft(0.9) {instance_size}{instance_type}",
        ),
        (
            "dgapq-cutoff-100phases-result_45642f3.json.gz",
            f"FDGA-Cutoff(10) {instance_size}{instance_type}",
        ),
    ]:
        plot_age_stats(pathlib.Path(base_dir) / file, title)


if __name__ == "__main__":
    instance_size = "medium"
    for instance_type in ["", "-fewerzones", "-fewzones", "-longbatch"]:
        # compute_tables(instance_size, instance_type)
        plot_examples(instance_size, instance_type)
