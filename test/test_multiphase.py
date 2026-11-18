import pytest
import batchingheuristics
from collections import Counter
import json
import gzip

import batchingheuristics.instances
import batchingheuristics.instances.definitions
import batchingheuristics.solvers


dist = {2: 100, 3: 50, 4: 20, 5: 5, 6: 2}  # distribution of order sizes
ioc_params = {
    "nbr_warehouse_items": 5_000,
    "nbr_articles": 1_000,
    "nbr_extra_items": 2_000,
    "nbr_zones": 3,
    "min_items_per_phase": 100,
    "max_orders_per_batch": 15,
    "init_orders": 300,
    "init_items": 2000,
}


def get_default_ioc():
    ioc = batchingheuristics.ItemOrderChain.initialize_from_parameters(
        dist,
        ioc_params["nbr_articles"],
        ioc_params["nbr_warehouse_items"],
        ioc_params["nbr_extra_items"],
        50,
        50,
        ioc_params["nbr_zones"],
    )
    return ioc


def test_IOC_generator():
    ioc = get_default_ioc()
    assert len(ioc.zones) == ioc_params["nbr_zones"]
    assert len(ioc.aisles) == 50 - 1  # specification includes aisle 0
    assert len(ioc.rows) == 50 - 1  # specification includes row 0
    assert (
        sum(len(group) for group in ioc.item_grouped_list)
        == ioc_params["nbr_warehouse_items"]
    )
    assert ioc.total_items_count == ioc_params["nbr_warehouse_items"]

    items, orders = ioc.initial_order_request(500, 50)
    assert len(items) == 500
    assert len(orders) == 50

    art_count_orders = Counter([a for o in orders for a in o.positions])
    order_size = sum(len(o.positions) for o in orders)
    art_count_items = Counter([i.article for i in items[:order_size]])
    assert art_count_items == art_count_orders

    items, orders = ioc.return_items_and_orders(500)
    assert len(items) == 500

    art_count_orders = Counter([a for o in orders for a in o.positions])
    order_size = sum(len(o.positions) for o in orders)
    art_count_items = Counter([i.article for i in items[:order_size]])
    assert art_count_items >= art_count_orders

    ioc.validate()


def test_IOC_errors():
    ioc = get_default_ioc()
    with pytest.raises(ValueError):
        ioc.initial_order_request(
            100, 100
        )  # Too few items requested to cover the number of orders

    ioc = get_default_ioc()
    with pytest.raises(ValueError):  # Not enough items available in IOC
        ioc.return_items_and_orders(10000)


def test_IOC_write(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    ioc_read = batchingheuristics.ItemOrderChain.initialize_from_file(
        tmp_path / "item-order-chain.json.gz"
    )

    assert ioc.item_grouped_list == ioc_read.item_grouped_list
    assert ioc.extra_items == ioc_read.extra_items
    assert ioc.order_list == ioc_read.order_list


def test_IOC_runner_write(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))
    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.RandomSolver, tmp_path
    )
    runner.run_init(50, 500)
    runner.run_phase(100)
    runner.run_phase(100)

    runner.write(tmp_path / "phases.json.gz")

    with gzip.open(tmp_path / "phases.json.gz", mode="rt") as fp:
        data = json.load(fp)
    assert len(data["phases"]) == 3


def test_multiphase_dgapq(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))

    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.DGAPQSolver, tmp_path
    )
    runner.run_init(50, 500, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=False)
    runner.run_phase(100, cvrp_routing=False)


def test_multiphase_inspection(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))

    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.DGAPQSolver, tmp_path
    )
    runner.run_init(50, 500, cvrp_routing=False)

    runner.get_agecount_orders_completed()
    runner.get_agecount_orders_open()
    runner.get_agecount_picked()
    runner.get_agecount_warehouse()
    runner.get_order_distances()
    runner.get_warehouse_distances()


def test_multiphase_cutoff_write(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))

    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.HardAntiAgingDGAPQSolverCutoff,
        tmp_path,
        max_age=20,
        n_new_orders=5,
    )
    runner.run_init(50, 500, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)

    out_path = tmp_path / "cutoff-phases.json.gz"
    runner.write(out_path)

    with gzip.open(out_path, mode="rt") as fp:
        data = json.load(fp)

    for phase in data["phases"]:
        for item_id in phase["warehouse_items"]:
            assert '"' not in item_id


def test_multiphase_hardantiaging(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))

    mandatory_age_completion = [30, 20, 10]
    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.HardAntiAgingDGAPQSolver,
        tmp_path,
        mandatory_age_completion=mandatory_age_completion,
    )
    runner.run_init(50, 500, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=False)
    runner.run_phase(100, cvrp_routing=False)


def test_multiphase_softantiaging(tmp_path):
    ioc = get_default_ioc()
    ioc.save_chain(tmp_path, "item-order-chain")

    params = batchingheuristics.instances.definitions.Parameters(
        min_number_requested_items=ioc_params["min_items_per_phase"],
        max_orders_per_batch=ioc_params["max_orders_per_batch"],
        max_container_volume=1000,
        first_row=min(ioc.rows),
        last_row=max(ioc.rows),
        first_aisle=min(ioc.aisles),
        last_aisle=max(ioc.aisles),
    )
    (tmp_path / "parameters.json").write_text(json.dumps(params.__dict__))

    runner = batchingheuristics.MultiphaseRunner(
        batchingheuristics.solvers.SoftAntiAgingDGAPQSolver, tmp_path
    )
    runner.run_init(50, 500, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=True)
    runner.run_phase(100, cvrp_routing=False)
    runner.run_phase(100, cvrp_routing=False)
