import numpy
from sys import getsizeof


class DistanceMatrix:
    def __init__(self):
        self.nodes = None

    def insert_node(self, node_object):
        pass


class Node:
    def __init__(self, link, value):
        self.link = link
        self.value = value

    def reset(self, new_link, new_value):
        self.link = new_link
        self.value = new_value






