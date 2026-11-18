import datetime
import itertools
import json
import math
import pathlib
import re
import statistics
from dataclasses import dataclass

import rich.console
import rich.table
from loguru import logger


def describe(data):
    if len(data) > 0:
        return {
            "min": min(data),
            "med": statistics.median(data),
            "mean": statistics.fmean(data),
            "max": max(data),
        }


class InstanceEncoder(json.JSONEncoder):
    def default(self, o):
        match o:
            case Article() | Parameters():
                return o.__dict__
            case WarehouseItem():
                ret = o.__dict__.copy()
                ret["article"] = o.article.id
                return ret
            case Order():
                ret = o.__dict__.copy()
                ret["positions"] = [pos.id for pos in o.positions]
                return ret
            case Batch():
                ret = o.__dict__.copy()
                ret["picklists"] = [
                    [item.id for item in picklist] for picklist in o.picklists
                ]
                ret["orders"] = [order.id for order in o.orders]
                return ret
            case _:
                return super().default(o)


@dataclass(frozen=True)
class Article:
    id: str
    volume: float

    def __repr__(self) -> str:
        return f"Article[{self.id}]"

    @classmethod
    def dummy(cls):
        return cls(id="Dummy Article", volume=0)


@dataclass(frozen=True)
class WarehouseItem:
    id: str
    article: Article
    row: int
    aisle: int
    zone: int

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self) -> str:
        return f"WarehouseItem[{self.id}](zone {self.zone}, row {self.row}, aisle {self.aisle}, article {self.article})"


@dataclass(frozen=True)
class Order:
    id: str
    positions: tuple[Article, ...]

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return f"Order[{self.id}]"


@dataclass(frozen=True)
class Parameters:
    min_number_requested_items: int
    max_orders_per_batch: int
    max_container_volume: int
    first_row: int
    last_row: int
    first_aisle: int
    last_aisle: int


class Batch:
    orders: list[Order]
    picklists: list[list[WarehouseItem]]

    def __init__(self, orders, picklists):
        self.picklists = picklists
        self.orders = orders

    def get_all_items(self):
        return list(itertools.chain(*self.picklists))

    def __repr__(self):
        return f"Batch({len(self.picklists)} picklists, {len(self.orders)} orders)"


def write_file_as_json(object_to_write, path):
    with open(path, "w") as f:
        f.write(json.dumps(object_to_write, indent=4, cls=InstanceEncoder))


class Instance:
    id: str
    articles: list[Article]
    orders: list[Order]
    warehouse_items: list[WarehouseItem]
    zones: list[int]
    parameters: Parameters
    batches: list[Batch]
    stats: dict

    def __init__(self) -> None:
        self.batches = []

    def write(self, directory):
        for object in ["articles", "warehouse_items", "orders", "parameters"]:
            write_file_as_json(
                self.__getattribute__(object), f"{directory}/{object}.json"
            )

    def store_result(self, path, algorithm):
        write_file_as_json(self.batches, f"{path}/batches-{algorithm}.json")
        self.stats["algorithm"] = algorithm
        write_file_as_json(self.stats, f"{path}/statistics-{algorithm}.json")

    def read(self, path=None):
        if path is None:
            path = self.id
        with open(f"{path}/parameters.json", "r") as file:
            self.parameters = Parameters(**json.load(file))

        with open(f"{path}/articles.json", "r") as file:
            self.articles = [Article(**a) for a in json.load(file)]

        articles_by_id = {a.id: a for a in self.articles}
        with open(f"{path}/orders.json", "r") as file:
            self.orders = [
                Order(
                    id=o["id"],
                    positions=tuple(articles_by_id[pos] for pos in o["positions"]),
                )
                for o in json.load(file)
            ]

        with open(f"{path}/warehouse_items.json", "r") as file:
            self.warehouse_items = []
            for w in json.load(file):
                w["article"] = articles_by_id[w["article"]]
                if isinstance(w["zone"], str):
                    # parse int from zone if given as string
                    numbers = re.findall(r"\d+", w["zone"])
                    w["zone"] = int(numbers[0]) if len(numbers) > 0 else -1

                self.warehouse_items.append(WarehouseItem(**w))

        self.zones = list({item.zone for item in self.warehouse_items})

    def get_article_item_dict(self):
        article_item = {}
        for item in self.warehouse_items:
            if item.article not in article_item:
                article_item[item.article] = [item]
            else:
                article_item[item.article].append(item)
        return article_item

    def get_zone_items_dict(self):
        zone_item = {}
        for item in self.warehouse_items:
            if item.zone not in zone_item:
                zone_item[item.zone] = [item]
            else:
                zone_item[item.zone].append(item)
        return zone_item

    def check_feasibility(self) -> bool:
        if (
            sum(len(picklist) for batch in self.batches for picklist in batch.picklists)
            < self.parameters.min_number_requested_items
        ):
            logger.error("Fewer items than requested")
            return False

        for batch in self.batches:
            if len(batch.orders) > self.parameters.max_orders_per_batch:
                logger.error("Batch exceeds max commissions limit!")
                return False

            articles = [
                article.id for order in batch.orders for article in order.positions
            ]
            picklist_articles = [
                item.article.id for picklist in batch.picklists for item in picklist
            ]
            if sorted(articles) != sorted(picklist_articles):
                logger.error(
                    "requested and assigned articles for some orders in this batch do not match!"
                )
                return False

            for picklist in batch.picklists:
                if len({item.zone for item in picklist}) > 1:
                    logger.error("picklist contains items of multiple zones!")
                    return False
                if (
                    sum(item.article.volume for item in picklist)
                    > self.parameters.max_container_volume
                ):
                    logger.error("Container volume exceeds limit")
                    return False

        # Assert that any item is only picked up once

        total_picklist_len = sum(
            sum(len(picklist) for picklist in batch.picklists) for batch in self.batches
        )

        item_id_set = set()

        for batch in self.batches:
            for picklist in batch.picklists:
                item_id_set.update([item.id for item in picklist])

        if total_picklist_len != len(item_id_set):
            logger.error("Some item is picked up more than once.")

        return True

    @staticmethod
    def aisle_distance(u: int, v: int):
        return abs(u - v)

    def row_distance(self, u: int, v: int, u_aisle: int, v_aisle: int):
        # Need to account for the case that the two items are in the same aisle
        if u_aisle == v_aisle:
            return abs(u - v)

        middle_distance = abs(u) + abs(v)
        if min(u, v) < 0 < max(u, v):
            return middle_distance
        elif u < 0:
            return min(
                middle_distance, 2 * abs(self.parameters.first_row) - middle_distance
            )
        else:
            return min(middle_distance, 2 * self.parameters.last_row - middle_distance)

    def distance(self, u: WarehouseItem, v: WarehouseItem):
        if u.zone != v.zone:
            return math.inf
        return self.row_distance(u.row, v.row, u.aisle, v.aisle) + self.aisle_distance(
            u.aisle, v.aisle
        )

    def distance_to_conveyor(self, u: WarehouseItem):
        conveyor_belt = WarehouseItem("conveyor", Article.dummy(), 0, 0, u.zone)
        return self.distance(u, conveyor_belt)

    def picklist_cost(self, picklist: list[WarehouseItem]) -> float | int:
        if len(picklist) == 0:
            return 0
        conveyor_belt = WarehouseItem(
            "conveyor", Article.dummy(), 0, 0, picklist[0].zone
        )
        return (
            self.distance(conveyor_belt, picklist[0])
            + self.distance(picklist[-1], conveyor_belt)
            + sum(
                self.distance(picklist[i], picklist[i + 1])
                for i in range(len(picklist) - 1)
            )
        )

    def batch_cost(self, batch: Batch) -> float | int:
        return sum(self.picklist_cost(pl) for pl in batch.picklists)

    def phase_cost(self, phase: list[Batch]) -> float | int:
        return sum(self.batch_cost(batch) for batch in phase)

    def evaluate(self, time_elapsed=None, print_table=False):
        if time_elapsed is not None:
            logger.info(f"Time elapsed: {time_elapsed:.2f}s")

        feasible = self.check_feasibility()
        if feasible:
            logger.info("Instance is feasible")
        else:
            logger.error("instance not feasible")

        nbr_picklist_items = sum(
            len(p) for batch in self.batches for p in batch.picklists
        )
        logger.info(f"number of picklist items: {nbr_picklist_items}")

        nbr_picklists = sum(len(batch.picklists) for batch in self.batches)
        logger.info(f"number of picklists: {nbr_picklists}")

        objective_value = sum(
            self.picklist_cost(picklist)
            for batch in self.batches
            for picklist in batch.picklists
        )
        logger.info(f"total distance: {objective_value}")

        instance_order_lengths = [len(order.positions) for order in self.orders]
        batched_order_lengths = [
            len(order.positions) for batch in self.batches for order in batch.orders
        ]

        self.stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "time_elapsed": time_elapsed,
            "nbr_picklist_items": nbr_picklist_items,
            "nbr_picklists": nbr_picklists,
            "objective_value": objective_value,
            "feasible": feasible,
            "picklist_cost_stats": describe(
                [
                    self.picklist_cost(picklist)
                    for batch in self.batches
                    for picklist in batch.picklists
                ]
            ),
            "avg_item_cost": objective_value / nbr_picklist_items,
            "instance_order_length_stats": describe(instance_order_lengths),
            "batched_order_length_stats": describe(batched_order_lengths),
            "picklist_length_stats": describe(
                [
                    len(picklist)
                    for batch in self.batches
                    for picklist in batch.picklists
                ]
            ),
            "picklist_container_level_stats": describe(
                [
                    sum(item.article.volume for item in picklist) / 1000
                    for batch in self.batches
                    for picklist in batch.picklists
                ]
            ),
        }

        if print_table:
            table = rich.table.Table(title="Run statistics")
            table.add_column("Name")
            table.add_column("Value")
            for key, value in self.stats.items():
                table.add_row(key, f"{value}")
            rich.console.Console().print(table)

    def read_batches(self, fname: str):
        items_by_id = {item.id: item for item in self.warehouse_items}
        orders_by_id = {order.id: order for order in self.orders}

        data = json.loads(pathlib.Path(fname).read_text())
        batches = []
        for batch in data:
            picklists = [
                [items_by_id[item] for item in picklist]
                for picklist in batch["picklists"]
            ]
            orders = [orders_by_id[order] for order in batch["orders"]]
            batches.append(Batch(orders, picklists))

        self.batches = batches
