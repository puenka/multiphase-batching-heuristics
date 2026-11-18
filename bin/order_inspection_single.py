import json
from collections import Counter
from pathlib import Path

from rich.progress import track

import batchingheuristics


def get_order_lengths(instance_dir, instance_name, algorithm):
    instance = batchingheuristics.Instance()
    instance.read(instance_dir / instance_name)
    instance.read_batches(
        instance_dir / instance_name / ("batches-" + algorithm + ".json")
    )
    instance_order_lengths = [len(order.positions) for order in instance.orders]
    batched_order_lengths = [
        len(order.positions) for batch in instance.batches for order in batch.orders
    ]
    avg_instance = sum(instance_order_lengths) / len(instance_order_lengths)
    avg_batches = sum(batched_order_lengths) / len(batched_order_lengths)
    return {
        "avg_instance_order_length": avg_instance,
        "avg_batched_order_length": avg_batches,
    }


def center_distances(instance_dir, instance_name, algorithm):
    instance = batchingheuristics.Instance()
    instance.read(instance_dir / instance_name)
    instance.read_batches(
        instance_dir / instance_name / ("batches-" + algorithm + ".json")
    )

    # compute average distance to conveyor for picked items
    avg_center_distance_batched = sum(
        [
            instance.distance_to_conveyor(item)
            for batch in instance.batches
            for pl in batch.picklists
            for item in pl
        ]
    ) / sum(
        len(order.positions) for batch in instance.batches for order in batch.orders
    )

    # compute average article distance for orders (lower bound), weighted by multiplicities
    article_counts = Counter()
    for order in instance.orders:
        for article in order.positions:
            article_counts[article] += 1

    a_i_dict = instance.get_article_item_dict()
    min_dist = {}
    for article in article_counts:
        min_dist[article] = min(
            instance.distance_to_conveyor(item) for item in a_i_dict[article]
        )

    avg_center_distance_orders_lowerbound = sum(
        min_dist[a] * m_a for a, m_a in article_counts.items()
    ) / sum(len(o.positions) for o in instance.orders)

    # compute average article distance for orders by picking the closest items first
    sorted_dists = {}
    for article in article_counts:
        sorted_dists[article] = sorted(
            instance.distance_to_conveyor(item) for item in a_i_dict[article]
        )

    avg_center_distance_orders_sorted = sum(
        sum(sorted_dists[a][:m_a]) for a, m_a in article_counts.items()
    ) / sum(len(o.positions) for o in instance.orders)

    return {
        "avg_batched_item_distance": avg_center_distance_batched,
        "avg_item_distance_lowerbound": avg_center_distance_orders_lowerbound,
        "avg_item_distance_sorted": avg_center_distance_orders_sorted,
    }


if __name__ == "__main__":
    instance_dir = Path("data/instances-singlephase/")

    for file in track(list(instance_dir.glob("*/statistics*"))):
        print(file)
        new_data = json.loads(file.read_text())
        new_data["instance"] = file.parent.name
        # new_data.update(get_order_lengths(instance_dir, new_data["instance"], new_data["algorithm"]))
        new_data.update(
            center_distances(instance_dir, new_data["instance"], new_data["algorithm"])
        )
        file.write_text(json.dumps(new_data))
