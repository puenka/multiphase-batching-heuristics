import random as r
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from batchingheuristics.instances.generator import generate_instance

parameters = {
    "debug": {"nbr_warehouse_items": 20, "nbr_orders": 5, "nbr_zones": 1},
    "tiny": {
        "nbr_warehouse_items": 500,
        "nbr_orders": 200,
        "nbr_zones": 3,
    },
    "small": {
        "nbr_warehouse_items": 10_000,
        "nbr_orders": 500,
        "nbr_zones": 10,
    },
    "medium": {
        "nbr_warehouse_items": 100_000,
        "nbr_orders": 5_000,
        "nbr_zones": 50,
    },
    "medium-longbatch": {
        "nbr_warehouse_items": 100_000,
        "nbr_orders": 5_000,
        "nbr_zones": 50,
        "max_orders_per_batch": 250,
    },
    "medium-highratio": {
        "nbr_warehouse_items": 100_000,
        "nbr_orders": 5_000,
        "nbr_zones": 50,
        "item_ratio": 15,
    },
    "medium-fewzones": {
        "nbr_warehouse_items": 100_000,
        "nbr_orders": 5_000,
        "nbr_zones": 10,
    },
    "medium-fewerzones": {
        "nbr_warehouse_items": 100_000,
        "nbr_orders": 5_000,
        "nbr_zones": 3,
    },
    "large": {
        "nbr_warehouse_items": 1_000_000,
        "nbr_orders": 50_000,
        "nbr_zones": 100,
    },
    "large-highratio": {
        "nbr_warehouse_items": 1_000_000,
        "nbr_orders": 50_000,
        "nbr_zones": 100,
        "item_ratio": 15,
    },
    "large-fewzones": {
        "nbr_warehouse_items": 1_000_000,
        "nbr_orders": 50_000,
        "nbr_zones": 10,
    },
    "large-fewerzones": {
        "nbr_warehouse_items": 1_000_000,
        "nbr_orders": 50_000,
        "nbr_zones": 3,
    },
    "large-longbatch": {
        "nbr_warehouse_items": 1_000_000,
        "nbr_orders": 50_000,
        "nbr_zones": 100,
        "max_orders_per_batch": 250,
    },
}


class InstanceSize(str, Enum):
    debug = "debug"
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"
    mediumfew = "medium-fewzones"
    mediumfewer = "medium-fewerzones"
    largefew = "large-fewzones"
    largefewer = "large-fewerzones"
    mediumratio = "medium-highratio"
    largeratio = "large-highratio"
    mediumbatch = "medium-longbatch"
    largebatch = "large-longbatch"


def generate_instances(
    sizes: list[InstanceSize],
    nbr_instances: Annotated[int, typer.Option("-n", "--nbr-instances")] = 5,
    dir: Annotated[Path, typer.Option("-d", "--dir", writable=True)] = Path(
        "data/instances"
    ),
    skip: Annotated[
        bool,
        typer.Option(
            help="Skip existing instances. --no-skip overwrites existing files."
        ),
    ] = True,
):
    r.seed(1)
    for size in sizes:
        logger.info(
            f"Creating {nbr_instances} instances of size {size.value} at '{dir}'."
        )
        for nbr in range(nbr_instances):
            path = dir / f"{size.value}-{nbr}"
            try:
                path.mkdir(parents=True, exist_ok=(not skip))
                generate_instance(path, **parameters[size.value])
            except FileExistsError:
                logger.warning(
                    f"Skipping instance creation because directory '{path}' already exists."
                )


if __name__ == "__main__":
    typer.run(generate_instances)
