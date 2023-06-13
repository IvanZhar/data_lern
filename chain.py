import random
import timeit


seq = [i for i in range(3000)]
random.Random(4).shuffle(seq)


class Node:
    def __init__(self, value, parent=None, child=None):
        self.value = value
        self.parent = parent
        self.child = child


class Chain:
    def __init__(self):
        self.root = None

    def add(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if val > node.value:
            if not node.parent:
                node.parent = Node(val, child=node)
            else:
                if val <= node.parent.value:
                    node.parent = Node(val, child=node, parent=node.parent)
                    node.parent.parent.child = node.parent
                else:
                    if node.parent.parent:
                        self._add(val, node.parent.parent)
                    else:
                        node.parent.parent = Node(val, child=node.parent)
        else:
            if node.child:
                node.child = Node(val, child=node.child, parent=node)
            else:
                self.root = Node(val, parent=node)
                node.child = self.root

    def print_chain(self):
        print('my_sort[', end='')
        if self.root:
            self._print_chain(self.root)
        print(']', end='')

    def _print_chain(self, node):
        if node.parent:
            print(f'{node.value}, ', end='')
            self._print_chain(node.parent)
        else:
            print(f'{node.value}', end='')


default = 'sorted(seq)'

print('default quicksort time: ', timeit.timeit(default, globals=globals(), number=100) / 100)


def my_sort(my_list):
    chain = Chain()
    for each in my_list:
        chain.add(each)


mine = 'my_sort(seq)'
print('mine chain sorting time: ', timeit.timeit(mine, globals=globals(), number=100) / 100)




# chain = Chain()
# chain.add(4)
# chain.add(2)
# chain.add(5)
# chain.add(3)
# chain.add(8)
# chain.add(12)
# chain.add(1)
# chain.add(6)
# chain.add(7)
# chain.add(0)
# chain.add(15)
# chain.print_chain()
