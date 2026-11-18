import random
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from loguru import logger

from batchingheuristics.instances.definitions import Instance
from batchingheuristics.solvers import (
    DGAPQSolver,
    NeighbourAisleSolver,
    NeighbourSolver,
    RandomSolver,
    ReferenceDGASolver,
    ReferenceRDGASampleSolver,
    ReferenceRDGASolver,
    ZGAFiveByNSolver,
    ZGAQuadSolver,
    ZGARadialSolver,
    ZGASolver,
)

Algorithm = Literal[
    "dga",
    "rdga",
    "random",
    "dgapq",
    "dgapq-skip",
    "rdga-5",
    "rdga-20",
    "rdga-50",
    "rdga-200",
    "rdga-sqrt",
    "zga",
    "zga-quad",
    "zga-radial",
    "zga-radial-quad",
    "zga-custom",
    "zga-fivebyfive",
    "neighbour",
    "neighbour-aisle",
]


def solve_instances(
    algorithm: Algorithm,
    instance_dir: Annotated[Path, typer.Argument(exists=True, writable=True)],
    cvrp_routing: Annotated[
        bool,
        typer.Option(
            help="Use a CVRP routine for routing. For --no-cvrp-routing, a simple heuristic is used."
        ),
    ] = True,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing files")] = False,
) -> None:
    """
    Compute batches for a warehouse batching/order selection/picker routing problem.

    The default file structure from the instance generator is expected. The directory of the instance should be given.

    If --cvrp-routing is enabled, the output of the algorithm will be stored by appending "cvrp" to the algorithm's
    name.
    """
    random.seed(1)

    target_fname = (
        instance_dir / f"batches-{algorithm}{'-cvrp' if cvrp_routing else ''}.json"
    )
    if target_fname.exists():
        if overwrite:
            logger.warning(
                f"Output file '{target_fname}' already exists and will be overwritten after the computation."
            )
        else:
            raise ValueError(
                f"Output file '{target_fname}' already exists. Use --overwrite to ignore existing files."
            )

    run(algorithm, instance_dir, cvrp_routing)


def run(algorithm, instance_path, cvrp_routing) -> None:
    logger.info(
        f"Running algorithm {algorithm} ({'cvrp' if cvrp_routing else 'heuristic'} routing) on instance '{instance_path}'"
    )
    logger.info("Reading instance")
    instance = Instance()
    instance.read(instance_path)
    logger.info("Creating batches")

    start_time = time.perf_counter()

    match algorithm:
        case "dgapq":
            solver = DGAPQSolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "dgapq-skip":
            solver = DGAPQSolver(instance, skip_article_distance_check=True)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "dga":
            solver = ReferenceDGASolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "random":
            instance.batches = RandomSolver(instance).compute_batches(cvrp_routing)
        case "rdga":
            solver = ReferenceRDGASolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "rdga-5":
            solver = ReferenceRDGASampleSolver(instance, 5)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "rdga-20":
            solver = ReferenceRDGASampleSolver(instance, 20)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "rdga-50":
            solver = ReferenceRDGASampleSolver(instance, 50)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "rdga-200":
            solver = ReferenceRDGASampleSolver(instance, 200)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "rdga-sqrt":
            solver = ReferenceRDGASampleSolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)

        case "zga":
            solver = ZGASolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "zga-quad":
            solver = ZGAQuadSolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "zga-radial":
            solver = ZGARadialSolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "zga-radial-quad":
            solver = ZGARadialSolver(instance, quad=True)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "zga-custom":
            target_count = 10
            solver = ZGAQuadSolver(instance, zonecount_mapping=target_count)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "zga-fivebyfive":
            solver = ZGAFiveByNSolver(instance, 5, zonecount_mapping=7)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "neighbour":
            solver = NeighbourSolver(instance)
            instance.batches = solver.compute_batches(cvrp_routing)
        case "neighbour-aisle":
            solver = NeighbourAisleSolver(instance, 100, 10)
            instance.batches = solver.compute_batches(cvrp_routing)
        case _:
            raise ValueError(f"Solver {algorithm} not recognized.")

    logger.info("batches created")
    time_elapsed = time.perf_counter() - start_time
    logger.info("Evaluating results")

    instance.evaluate(time_elapsed, print_table=True)
    logger.info("writing results")
    instance.store_result(instance_path, algorithm + ("-cvrp" if cvrp_routing else ""))
    logger.info(f"Results for {algorithm} on '{instance_path}' computed and stored.")


if __name__ == "__main__":
    typer.run(solve_instances)
