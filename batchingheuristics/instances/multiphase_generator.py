import collections
import gzip
import json
import logging
import pathlib
import random as r

from numpy.random import default_rng

from batchingheuristics.instances.definitions import (
    Article,
    InstanceEncoder,
    Order,
    WarehouseItem,
)

logger = logging.getLogger(__name__)


class ItemOrderChain:
    """
    This class represents a multiphase instance. It can be used for an initial warehouse state with a corresponding set
    of valid orders and excess items, as well as for restocking it for subsequent phases by items from new orders.
    """

    item_grouped_list: list[list[WarehouseItem]]
    order_list: list[Order]
    total_items_count: int
    extra_items: list[WarehouseItem]

    articles: list[Article]

    aisles: list[int]
    rows: list[int]
    zones: list[int]

    location_queue: collections.deque[WarehouseItem]

    def __init__(self):
        self.articles = []
        self.item_grouped_list = []
        self.total_items_count = 0
        self.order_list = []
        self.extra_items = []
        self.aisles = []
        self.rows = []
        self.zones = []
        self.location_queue = collections.deque()

    @classmethod
    def initialize_from_parameters(
        cls,
        order_size_dist,
        num_articles: int,
        num_items: int,
        num_extra_items: int,
        num_aisles: int,
        num_rows: int,
        num_zones: int,
    ):
        """Initialize the class by creating a new multiphase instance with the given parameters."""
        ioc = cls()
        ioc.articles = ioc.generate_articles(num_articles)
        ioc.zones = list(range(num_zones))
        ioc.aisles = ioc.generate_warehouse_aisles_or_rows(num_aisles)
        ioc.rows = ioc.generate_warehouse_aisles_or_rows(num_rows)
        ioc.item_grouped_list, ioc.order_list = ioc.generate_item_order_chain(
            order_size_dist, num_items
        )
        ioc.total_items_count = sum(len(sublist) for sublist in ioc.item_grouped_list)
        ioc.extra_items = ioc.generate_additional_items(num_extra_items)

        return ioc

    @classmethod
    def initialize_from_file(cls, filename):
        """Initialize the class by load an existing multiphase instance."""
        ioc = cls()
        f_path = pathlib.Path(filename)

        with gzip.open(f_path, mode="rt") as f:
            (
                item_grouped_list_dicts,
                order_list_dicts,
                ioc.total_items_count,
                extra_items_dicts,
                articles_dicts,
                ioc.aisles,
                ioc.rows,
                ioc.zones,
            ) = json.load(f)

        article_by_id = {}
        item_by_id = {}

        # Article and item references are stored as strings in the data, so we need to assign the corresponding objects
        for article in articles_dicts:
            a = Article(**article)
            ioc.articles.append(a)
            article_by_id[a.id] = a

        for group in item_grouped_list_dicts:
            group_items = []
            for item in group:
                item["article"] = article_by_id[item["article"]]
                i = WarehouseItem(**item)
                item_by_id[i.id] = i
                group_items.append(i)
            ioc.item_grouped_list.append(group_items)

        for order in order_list_dicts:
            o = Order(
                order["id"], tuple(article_by_id[a_id] for a_id in order["positions"])
            )
            ioc.order_list.append(o)

        assert ioc.total_items_count == sum(map(len, ioc.item_grouped_list))

        for item in extra_items_dicts:
            item["article"] = article_by_id[item["article"]]
        ioc.extra_items = [WarehouseItem(**item) for item in extra_items_dicts]

        return ioc

    def initial_order_request(
        self, item_number: int, order_number: int
    ) -> tuple[list[WarehouseItem], list[Order]]:
        """
        Returns items and orders for an initial request. Assumes that #itemsrequested is (much) higher than the
        specified order amount, so will first take items and orders from item chain until number of orders is fulfilled
        and then fill with newly generated items.
        """

        remaining_items_requested = item_number
        requested_items = []
        requested_orders = []
        for _ in range(order_number):
            new_items = self.item_grouped_list.pop()
            if remaining_items_requested < len(new_items):
                raise ValueError(
                    "The number of requested items is not large enough to cover the requested number of orders"
                )
            remaining_items_requested -= len(new_items)
            requested_items.extend(new_items)
            requested_orders.append(self.order_list.pop())

        self.total_items_count -= len(requested_items)

        # if there are items left to be added, pop them from the extra item list
        remaining_count = item_number - len(requested_items)
        if remaining_count > 0:
            assert len(self.extra_items) >= remaining_count, (
                "Not enough extra items in IOC to fulfill request."
            )
            requested_items.extend(self.extra_items[:remaining_count])
            del self.extra_items[:remaining_count]

        return requested_items, requested_orders

    def return_items_and_orders(
        self, num_items_requested
    ) -> tuple[list[WarehouseItem], list[Order]]:
        """
        Returns the requested number of items and all orders which are fully covered by the items (so the last order
        might not be included)
        """
        if num_items_requested > self.total_items_count:
            raise ValueError("Not enough items remaining to fulfill request.")
        requested_items = []
        requested_orders = []

        num_items_requested_iter = num_items_requested

        while num_items_requested_iter >= len(self.item_grouped_list[-1]):
            new_items = self.item_grouped_list.pop()
            for item in new_items:
                if (
                    len(self.location_queue) > 0
                ):  # use location from queue instead of original item location
                    loc_item = self.location_queue.popleft()
                    item = WarehouseItem(
                        item.id,
                        item.article,
                        loc_item.row,
                        loc_item.aisle,
                        loc_item.zone,
                    )

            requested_items.extend(new_items)
            requested_orders.append(self.order_list.pop())
            num_items_requested_iter -= len(new_items)

        if num_items_requested_iter > 0:
            requested_items.extend(
                self.item_grouped_list[-1][:num_items_requested_iter]
            )
            self.item_grouped_list[-1] = self.item_grouped_list[-1][
                num_items_requested_iter:
            ]

        self.total_items_count -= num_items_requested
        assert len(requested_items) == num_items_requested

        return requested_items, requested_orders

    def extend_restocking_locations(self, items: list[WarehouseItem]):
        """
        Appends the given items to the FIFO queue of locations onto which newly requested items will be restocked to.
        """
        self.location_queue.extend(items)

    def generate_item_order_chain(
        self, dist, num_items
    ) -> tuple[list[list[WarehouseItem]], list[Order]]:
        """
        Generates list of items (of specified length) and "chops" it into orders, returns list of lists of items and
        list of orders, where each sublist corresponds to the items "belonging" to the order at the resp. index
        """
        # Generate list of items, ensuring that each article is represented once
        item_base_chain = []
        for i in range(num_items):
            aisle = r.choice(self.aisles)
            row = r.choice(self.rows)
            zone = r.choice(self.zones)

            if i < len(self.articles):
                article = self.articles[i]
            else:
                article = r.choice(self.articles)
            item_base_chain.append(
                WarehouseItem(f"warehouse-item-{i}", article, row, aisle, zone)
            )
        r.shuffle(item_base_chain)

        # Generate list of orders and respectively grouped items by order
        order_list = []
        item_grouped_list = []
        index = -1
        item_index = 0
        # Generate list from distribution
        com_items_distribution = list(collections.Counter(dist).elements())
        while item_index < len(item_base_chain):
            index += 1
            nbr_positions = min(
                len(item_base_chain) - item_index, r.choice(com_items_distribution)
            )
            next_items = item_base_chain[item_index : (item_index + nbr_positions)]
            next_order = Order(
                f"order-{index}", tuple(item.article for item in next_items)
            )
            order_list.append(next_order)
            item_grouped_list.append(next_items)
            item_index += nbr_positions

        return item_grouped_list, order_list

    def generate_additional_items(self, num_items):
        """
        Generates the given number of new items, used at the start for initializing with additional items.
        Additional items get 100_000_000_000 added to their id number.

        """

        rng = default_rng()

        aisles = rng.choice(self.aisles, size=num_items)
        rows = rng.choice(self.rows, size=num_items)
        zones = rng.choice(self.zones, size=num_items)
        articles = rng.choice(self.articles, size=num_items)  # pyright: ignore[reportArgumentType, reportCallIssue]

        additional_items = [
            WarehouseItem(
                f"warehouse-item-{100_000_000_000 + i}",
                article,
                row.item(),
                aisle.item(),
                zone.item(),
            )
            for i, (row, aisle, article, zone) in enumerate(
                zip(aisles, rows, articles, zones)
            )
        ]
        return additional_items

    def save_chain(self, directory, fname):
        """
        Save the instance to disk. The directory will be created if it doesn't exist yet.
        The instance is saved to 'directory/fname.json.gz' as compressed json data.
        """
        obj_to_write = (
            self.item_grouped_list,
            self.order_list,
            self.total_items_count,
            self.extra_items,
            self.articles,
            self.aisles,
            self.rows,
            self.zones,
        )
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(exist_ok=True)
        with gzip.open(
            dir_path / (fname + ".json.gz"), mode="wt", encoding="utf-8"
        ) as f:
            json.dump(obj_to_write, f, indent=4, cls=InstanceEncoder)

    @staticmethod
    def generate_warehouse_aisles_or_rows(num):
        aisles_or_rows = list(range(-(num - 1) // 2, num // 2))
        aisles_or_rows.remove(0)  # no items on aisles/row 0
        return aisles_or_rows

    @staticmethod
    def generate_articles(num_articles):
        return [
            Article(f"article-{i}", round(r.gammavariate(2.0, 20.0)))
            for i in range(num_articles)
        ]

    def validate(self):
        assert self.total_items_count == sum(map(len, self.item_grouped_list)), (
            "Item counter is offset."
        )

        # the last order could be only partially covered because some items have already been taken
        for order, item_group in zip(self.order_list[:-1], self.item_grouped_list[:-1]):
            assert set(order.positions) == {item.article for item in item_group}, (
                "An order does not correspond to its item group"
            )
