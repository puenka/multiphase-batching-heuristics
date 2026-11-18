from batchingheuristics import ItemOrderChain
from batchingheuristics.instances.definitions import Parameters
import typer
from enum import Enum
from typing import Annotated, Literal
from pathlib import Path
from loguru import logger
import json


parameters = {
    "tiny": {
        "nbr_warehouse_items": 5_000,
        "nbr_articles": 1_000,
        "nbr_zones": 3,
        "min_items_per_phase": 100,
        "max_orders_per_batch": 15,
        "init_orders": 300,
        "init_items": 2000,
    },
    "small": {
        "nbr_warehouse_items": 30_000,
        "nbr_articles": 3_333,
        "nbr_zones": 10,
        "min_items_per_phase": 500,
        "max_orders_per_batch": 30,
        "init_orders": 500,
        "init_items": 10_000,
    },
    "medium": {
        "nbr_warehouse_items": 300_000,  # should cover 100*min_items_per_phase, it's the full stock, not just a single phase
        "nbr_articles": 33_333,  # 1/3 of warehouse items (init_items) of single-phase instance (implicitly generated there)
        "nbr_zones": 50,
        "min_items_per_phase": 2_500,  # ~ 1000 orders * average order size (actually a bit more in the single-phase)
        "max_orders_per_batch": 50,
        "init_orders": 5_000,
        "init_items": 100_000,
    },
    "medium-longbatch": {
        "nbr_warehouse_items": 300_000,  # should cover 100*min_items_per_phase, it's the full stock, not just a single phase
        "nbr_articles": 33_333,  # 1/3 of warehouse items (init_items) of single-phase instance (implicitly generated there)
        "nbr_zones": 50,
        "min_items_per_phase": 2_500,  # ~ 1000 orders * average order size (actually a bit more in the single-phase)
        "max_orders_per_batch": 250,
        "init_orders": 5_000,
        "init_items": 100_000,
    },
    "medium-fewzones": {
        "nbr_warehouse_items": 300_000,  # should cover 100*min_items_per_phase, it's the full stock, not just a single phase
        "nbr_articles": 33_333,  # 1/3 of warehouse items (init_items) of single-phase instance (implicitly generated there)
        "nbr_zones": 10,
        "min_items_per_phase": 2_500,  # ~ 1000 orders * average order size (actually a bit more in the single-phase)
        "max_orders_per_batch": 50,
        "init_orders": 5_000,
        "init_items": 100_000,
    },
    "medium-fewerzones": {
        "nbr_warehouse_items": 300_000,  # should cover 100*min_items_per_phase, it's the full stock, not just a single phase
        "nbr_articles": 33_333,  # 1/3 of warehouse items (init_items) of single-phase instance (implicitly generated there)
        "nbr_zones": 3,
        "min_items_per_phase": 2_500,  # ~ 1000 orders * average order size (actually a bit more in the single-phase)
        "max_orders_per_batch": 50,
        "init_orders": 5_000,
        "init_items": 100_000,
    },
    "large": {  # about 10x medium
        "nbr_warehouse_items": 3_000_000,
        "nbr_articles": 333_333,
        "nbr_zones": 100,
        "min_items_per_phase": 25_000,
        "max_orders_per_batch": 50,
        "init_orders": 50_000,
        "init_items": 1_000_000,
    },
    "large-fewzones": {  # about 10x medium
        "nbr_warehouse_items": 3_000_000,
        "nbr_articles": 333_333,
        "nbr_zones": 10,
        "min_items_per_phase": 25_000,
        "max_orders_per_batch": 50,
        "init_orders": 50_000,
        "init_items": 1_000_000,
    },
    "large-fewerzones": {  # about 10x medium
        "nbr_warehouse_items": 3_000_000,
        "nbr_articles": 333_333,
        "nbr_zones": 3,
        "min_items_per_phase": 25_000,
        "max_orders_per_batch": 50,
        "init_orders": 50_000,
        "init_items": 1_000_000,
    },
}

InstanceSize = Literal[
    "tiny",
    "small",
    "medium",
    "medium-longbatch",
    "medium-fewzones",
    "medium-fewerzones",
    "large",
    "large-fewzones",
    "large-fewerzones",
]


def generate_multiphase(
    size: InstanceSize,
    nbr_instances: Annotated[int, typer.Option("-n", "--nbr-instances")] = 5,
    dir: Annotated[Path, typer.Option("-d", "--dir", writable=True)] = Path(
        "data/instances-multiphase"
    ),
    skip: Annotated[
        bool,
        typer.Option(
            help="Skip existing instances. --no-skip overwrites existing files."
        ),
    ] = True,
):
    dist = {2: 100, 3: 50, 4: 20, 5: 5, 6: 2}  # distribution of order sizes

    logger.info(f"Creating {nbr_instances} instances of size {size} at '{dir}'.")
    for nbr in range(nbr_instances):
        instance_path = dir / f"{size}-{nbr}"
        try:
            instance_path.mkdir(parents=True, exist_ok=(not skip))
            IOC = ItemOrderChain.initialize_from_parameters(
                dist,
                num_articles=parameters[size]["nbr_articles"],
                num_items=parameters[size]["nbr_warehouse_items"],
                num_aisles=100,
                num_rows=100,
                num_zones=parameters[size]["nbr_zones"],
                num_extra_items=parameters[size]["init_items"],
            )
            IOC.save_chain(instance_path, "item-order-chain")
            params = Parameters(
                min_number_requested_items=parameters[size]["min_items_per_phase"],
                max_orders_per_batch=parameters[size]["max_orders_per_batch"],
                max_container_volume=1000,
                first_row=min(IOC.rows),
                last_row=max(IOC.rows),
                first_aisle=min(IOC.aisles),
                last_aisle=max(IOC.aisles),
            )
            (instance_path / "parameters.json").write_text(json.dumps(params.__dict__))
            (instance_path / "init_config.json").write_text(
                json.dumps(
                    {
                        "init_orders": parameters[size]["init_orders"],
                        "init_items": parameters[size]["init_items"],
                    }
                )
            )
        except FileExistsError:
            logger.warning(
                f"Skipping instance creation because directory '{instance_path}' already exists."
            )
        else:
            logger.info(f"Instance has been written to '{instance_path}'")


if __name__ == "__main__":
    typer.run(generate_multiphase)
