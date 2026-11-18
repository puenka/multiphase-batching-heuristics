"""A class implementing a min heap priority queue, based on the implementation from the heapq module, found at
https://github.com/python/cpython/blob/3.12/Lib/heapq.py. Does not implement all methods, mainly initializing and
popping min element, but enhances with being able to update the cost of individual elements. Achieves this by tracking
item indices during heap operations, this incurs some cost overhead."""

import random


class MinPQ:
    heap: list
    item_index_dict: dict[any, int]

    def __init__(self, initial_set):
        self.item_index_dict = {}
        self.heap = list(initial_set)
        self.heapify(initial_set)

    def __len__(self):
        return len(self.heap)

    def heappush(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        self.heap.append(item)
        self.item_index_dict[item] = len(self.heap) - 1
        self._siftdown(0, len(self.heap) - 1)

    def heappop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = self.heap.pop()  # raises appropriate IndexError if heap is empty
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self.item_index_dict[lastelt] = 0
            self._siftup(0)
            return returnitem
        return lastelt

    def heappeek(self):
        return self.heap[0]

    def heapify(self, x):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(x)
        self.heap = x
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            self._siftup(i)

        # Create dictionary
        self._initialize_item_dict()

    def update_item(self, item, new_value):
        index = self.item_index_dict[item]
        if self.heap[index] < new_value:
            self.heap[index] = new_value
            self._siftup(index)
        else:
            self.heap[index] = new_value
            self._siftdown(0, index)
        if item != new_value:
            self.item_index_dict.pop(item)

    def _initialize_item_dict(self):
        ind = 0
        for item in self.heap:
            self.item_index_dict[item] = ind
            ind += 1

    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                self.item_index_dict[parent] = pos
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem
        self.item_index_dict[newitem] = pos

    def _siftup(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            self.item_index_dict[self.heap[childpos]] = pos
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self.item_index_dict[newitem] = pos
        self._siftdown(startpos, pos)
