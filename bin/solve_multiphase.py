from batchingheuristics import MultiphaseRunner
from batchingheuristics.solvers import (
    DGAPQSolver,
    SoftAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolver,
    HardAntiAgingDGAPQSolverLinear,
    HardAntiAgingDGAPQSolverCutoff,
    ReferenceRDGASampleSolver,
    NeighbourAisleSolver,
)
from loguru import logger
import typer
from collections import Counter
from typing import Annotated, Literal
from pathlib import Path
import json
import hashlib

Algorithm = Literal[
    "dgapq",
    "neighbour-aisle",
    "rdga-sqrt",
    "dgapq-soft",
    "dgapq-hard",
    "dgapq-linear",
    "dgapq-cutoff",
]


def needs_params(algo: Algorithm):
    return algo in ["dgapq-soft", "dgapq-hard", "dgapq-linear", "dgapq-cutoff"]


def solve_multiphase(
    algorithm: Algorithm,
    instance_path: Annotated[Path, typer.Argument(exists=True, writable=True)],
    n_phases: int = 100,
    init_config_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to json with initialization data. Default: 'instance_path/init_config.json'",
            exists=True,
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing files")] = False,
    solver_params_path: Annotated[Path | None, typer.Option(exists=True)] = None,
    solver_name: Annotated[
        str | None, typer.Option(help="Descriptive string for the results file")
    ] = None,
    skip_restocking_every: Annotated[
        int,
        typer.Option(
            help="Skip restocking every n-th phase (n > 1). Skipping is disabled for 0"
        ),
    ] = 0,
):
    # Prepare
    if needs_params(algorithm) and solver_params_path is None:
        raise ValueError(
            f"Please provide a file containing solver parameters for {algorithm} with '--solver-params-path'"
        )

    logger.info(
        f"Loading runner on instance {instance_path}. Solver {algorithm} (name {solver_name}). Params file {solver_params_path}."
    )

    solver_params = dict()

    match algorithm:
        case "dgapq":
            solver = DGAPQSolver
        case "neighbour-aisle":
            solver = NeighbourAisleSolver
        case "dgapq-soft":
            solver_params = json.loads(solver_params_path.read_text())  # type: ignore
            solver = SoftAntiAgingDGAPQSolver
        case "dgapq-hard":
            age_mapping = json.loads(solver_params_path.read_text())  # type: ignore
            # convert keys to int
            solver_params = {
                "mandatory_age_completion": {int(k): v for k, v in age_mapping.items()}
            }
            solver = HardAntiAgingDGAPQSolver
        case "dgapq-linear":
            solver_params = json.loads(solver_params_path.read_text())  # type: ignore
            solver = HardAntiAgingDGAPQSolverLinear
        case "dgapq-cutoff":
            solver_params = json.loads(solver_params_path.read_text())  # type: ignore
            solver = HardAntiAgingDGAPQSolverCutoff
        case "rdga-sqrt":
            solver = ReferenceRDGASampleSolver
        case _:  # redundant argument check
            raise ValueError(f"Solver {algorithm} not recognized.")

    if solver_name:
        target_fname_base = f"{solver_name}-{n_phases}phases-result"
    else:
        target_fname_base = f"{algorithm}-{n_phases}phases-result"

    if needs_params(algorithm):
        if "id" in solver_params.keys():
            target_fname = (
                target_fname_base + "_" + str(solver_params["id"]) + ".json.gz"
            )
            solver_params.pop("id")
        else:
            params_hash = hashlib.md5(solver_params_path.read_bytes()).hexdigest()[:7]  # type: ignore
            target_fname = target_fname_base + "_" + params_hash + ".json.gz"
    else:
        target_fname = target_fname_base + ".json.gz"

    if (instance_path / target_fname).exists():
        if overwrite:
            logger.warning(
                f"Output file '{target_fname}' already exists and will be overwritten after the computation."
            )
        else:
            raise ValueError(
                f"Output file '{target_fname}' already exists. Use --overwrite to ignore existing files."
            )

    # Run
    runner = MultiphaseRunner(solver, instance_path, restocking=False, **solver_params)
    runner.item_order_chain.validate()

    if init_config_path is None:
        init_config = json.loads((instance_path / "init_config.json").read_text())
    else:
        init_config = json.loads(init_config_path.read_text())
    logger.info(f"Starting initial phase with solver {solver.__name__}.")
    logger.info(
        f"The initial warehouse has {init_config['init_orders']} orders and {init_config['init_items']} items."
    )

    runner.run_init(
        init_config["init_orders"], init_config["init_items"], cvrp_routing=True
    )

    ages_per_phase = [runner.get_agecount_warehouse()]

    for i in range(n_phases):
        logger.info(f"Starting phase {i + 1}.")
        runner.item_order_chain.validate()
        logger.debug(runner)
        num_items_removed = sum(
            [len(o.positions) for b in runner.phases[-1].batches for o in b.orders]
        )

        if skip_restocking_every == 0:
            runner.run_phase(
                num_items=num_items_removed, cvrp_routing=True
            )  # Add as many items as needed to keep the number of items in the warehouse constant
        else:
            if (
                i + 1
            ) % skip_restocking_every == 0:  # shift by 1 to start with restocking
                logger.info("Running phase for old orders without restocking")
                runner.run_phase_oldest(cvrp_routing=True)
            else:
                logger.info("Running phase with restocking")
                runner.run_phase(
                    num_items=round(
                        num_items_removed * (1 + 1 / (skip_restocking_every - 1))
                    ),
                    cvrp_routing=True,
                )  # Add new orders for every other phase, adjusting for the removed items in the previous phase and adding fractionally more to compensate for the skipped phases

        logger.info(f"Phase {i + 1} finished. Saving age data.")
        ages_per_phase.append(runner.get_agecount_warehouse())
        logger.info(Counter([runner.order_ages[order] for order in runner.orders]))

    if needs_params(algorithm):
        runner.write(instance_path / target_fname, params=solver_params)
    else:
        runner.write(instance_path / target_fname)


if __name__ == "__main__":
    typer.run(solve_multiphase)
