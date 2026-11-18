import json

import pytest

import batchingheuristics
import batchingheuristics.instances
import batchingheuristics.instances.definitions
import batchingheuristics.instances.generator
import batchingheuristics.solvers

instance_params = {"nbr_warehouse_items": 500, "nbr_orders": 200, "nbr_zones": 3}


def test_instance_generation_to_file(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.instances.definitions.Instance()
    instance.read(tmp_path)
    assert len(instance.warehouse_items) == instance_params["nbr_warehouse_items"]
    assert len(instance.orders) == instance_params["nbr_orders"]
    assert len(instance.zones) == instance_params["nbr_zones"]


def test_instance_members(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.instances.definitions.Instance()
    instance.read(tmp_path)

    art_item = instance.get_article_item_dict()
    for a, a_items in art_item.items():
        assert all(i.article == a for i in a_items)

    zone_items_dict = instance.get_zone_items_dict()
    for zone, zone_items in zone_items_dict.items():
        assert all(i.zone == zone for i in zone_items)


def test_instance_feasibility_randomsolver(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.instances.definitions.Instance()
    instance.read(tmp_path)

    instance.batches = batchingheuristics.solvers.RandomSolver(
        instance
    ).compute_batches(cvrp_routing=False)
    assert instance.check_feasibility()
    instance.evaluate()


def test_instance_basicsolvers(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.instances.definitions.Instance()
    instance.read(tmp_path)

    for solver in [
        batchingheuristics.solvers.DGAPQSolver(instance),
        batchingheuristics.solvers.RandomSolver(instance),
        batchingheuristics.solvers.ReferenceDGASolver(instance),
    ]:
        instance.batches = solver.compute_batches()
        assert instance.check_feasibility()

    solver = batchingheuristics.solvers.DGAPQSolver(instance)
    instance.batches = solver.compute_batches(cvrp_routing=False)
    assert instance.check_feasibility()


def test_instance_solution_storage(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.Instance()
    instance.read(tmp_path)

    instance.batches = batchingheuristics.solvers.RandomSolver(
        instance
    ).compute_batches(cvrp_routing=False)
    instance.evaluate()
    instance.store_result(tmp_path, "test")
    batches_ref = instance.batches
    instance.read_batches(tmp_path / "batches-test.json")

    stats = json.loads((tmp_path / "statistics-test.json").read_text())
    assert stats["algorithm"] == "test"
    pytest.approx(
        stats["instance_order_length_stats"]["mean"],
        sum(len(o.positions) for o in instance.orders) / len(instance.orders),
    )

    for batch_disk, batch in zip(instance.batches, batches_ref):
        assert batch.orders == batch_disk.orders
        assert batch.picklists == batch_disk.picklists


def test_algos_greedy(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.Instance()
    instance.read(tmp_path)

    for solver in [
        batchingheuristics.solvers.ReferenceDGASolver(instance),
        batchingheuristics.solvers.ReferenceRDGASolver(instance),
        batchingheuristics.solvers.RandomSolver(instance),
        batchingheuristics.solvers.DGAPQSolver(instance),
        batchingheuristics.solvers.DGAPQSolver(
            instance, skip_article_distance_check=True
        ),
        batchingheuristics.solvers.ReferenceRDGASampleSolver(instance),
        batchingheuristics.solvers.ReferenceRDGASampleSolver(instance, 5),
        batchingheuristics.solvers.ZGASolver(instance),
        batchingheuristics.solvers.ZGAQuadSolver(instance),
        batchingheuristics.solvers.ZGAFiveByNSolver(instance),
        batchingheuristics.solvers.ZGARadialSolver(instance),
        batchingheuristics.solvers.NeighbourSolver(instance),
        batchingheuristics.solvers.NeighbourAisleSolver(instance),
    ]:
        instance.batches = solver.compute_batches(cvrp_routing=False)
        assert instance.check_feasibility()


def test_algos_cvrp(tmp_path):
    batchingheuristics.instances.generator.generate_instance(
        tmp_path, **instance_params
    )
    instance = batchingheuristics.instances.definitions.Instance()
    instance.read(tmp_path)

    for solver in [
        batchingheuristics.solvers.ReferenceDGASolver(instance),
        batchingheuristics.solvers.ReferenceRDGASolver(instance),
        batchingheuristics.solvers.RandomSolver(instance),
        batchingheuristics.solvers.DGAPQSolver(instance),
        batchingheuristics.solvers.ReferenceRDGASampleSolver(instance),
        batchingheuristics.solvers.ReferenceRDGASampleSolver(instance, 5),
        batchingheuristics.solvers.ZGASolver(instance),
        batchingheuristics.solvers.ZGAQuadSolver(instance),
        batchingheuristics.solvers.NeighbourSolver(instance),
        batchingheuristics.solvers.NeighbourAisleSolver(instance),
    ]:
        instance.batches = solver.compute_batches(cvrp_routing=True)
        assert instance.check_feasibility()
