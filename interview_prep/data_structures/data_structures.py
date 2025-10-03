# Data structures in Python:
# We will implement the following data structures in Python:
# 1. List
# 2. Tuple
# 3. Set
# 4. Dictionary
# 5. Stack
# 6. Queue
# 7. Linked List
# 8. Binary Tree
# 9. Graph
# 10. Heap
# 11. Red-Black Tree
# 12. AVL Tree
# 13. B-Tree


# 1. List
# A list is a collection which is ordered and changeable. In Python lists are written with square brackets.
# When to use: Use a list when you have a collection of data that does not need random access. Lists are iterable.
# Example:
# Define a list:
class PyList:
    """
    Python List is a dynamic array.
    Operations:
    - Append: O(1) on average.
    - Pop: O(1) from the end, O(n) from the beginning or middle.
    - Access: O(1) time complexity.
    Use Case: When you need a simple, flexible collection that can grow dynamically.
    """
    def __init__(self, iterable=[]):
        self.items = list(iterable)

    def append(self, item):
        self.items.append(item)
    
    def pop(self, index=-1):
        if self.is_empty():
            raise IndexError("pop from empty list")
        return self.items.pop(index)
    
    def is_empty(self):
        return len(self.items) == 0
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)
    
    def __str__(self):
        return str(self.items)
# Create a list:
fruits = PyList(["apple", "banana", "cherry"])
print(fruits)
# Access items:
print(fruits[1])
# Change item value:
fruits[1] = "blackcurrant"
print(fruits)
# Loop through a list:
for x in fruits:
    print(x)
# Check if item exists:
if "apple" in fruits:
    print("Yes, 'apple' is in the fruits list")
# List length:
print(len(fruits))
# Add items:
fruits.append("orange")
print(fruits)
# Remove item:
fruits.remove("banana")
print(fruits)
# Remove item by index:
fruits.pop(1)
print(fruits)
# Copy a list:
fruits_copy = fruits.copy()
print(fruits_copy)
# Join two lists:
vegetables = ["carrot", "potato", "onion"]
fruits_and_vegetables = fruits + vegetables
print(fruits_and_vegetables)
# Clear a list:
fruits.clear()
print(fruits)

# 2. Tuple
# A tuple is a collection which is ordered and unchangeable. In Python tuples are written with round brackets.
# When to use: Use a tuple when you have a collection of data that does not need to change. Tuples are iterable.
# Example:
# Create a tuple:
fruits = ("apple", "banana", "cherry")
print(fruits)
# Access items:
print(fruits[1])
# Loop through a tuple:
for x in fruits:
    print(x)
# Check if item exists:
if "apple" in fruits:
    print("Yes, 'apple' is in the fruits tuple")
# Tuple length:
print(len(fruits))
# Add items:
# You cannot add items to a tuple. Tuples are unchangeable.
# Remove items:
# You cannot remove items in a tuple. Tuples are unchangeable.
# Join two tuples:
vegetables = ("carrot", "potato", "onion")
fruits_and_vegetables = fruits + vegetables
print(fruits_and_vegetables)
# The tuple() Constructor:
fruits = tuple(("apple", "banana", "cherry"))  # note the double round-brackets
print(fruits)

# 3. Set
# A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.
# When to use: Use a set when you want to store a collection of values without duplicates. Sets are iterable.
# Example:
# Create a set:
fruits = {"apple", "banana", "cherry"}
print(fruits)
# Access items:
# You cannot access items in a set by referring to an index, since sets are unordered the items has no index.
# But you can loop through the set items using a for loop, or ask if a specified value is present in a set, by using the in keyword.
# Loop through a set:
for x in fruits:
    print(x)
# Check if item exists:
print("banana" in fruits)
# Add items:
fruits.add("orange")
print(fruits)
# Add multiple items:
fruits.update(["orange", "mango", "grapes"])
print(fruits)
# Get the length of a set:
print(len(fruits))
# Remove item:
fruits.remove("banana")
print(fruits)
# Remove item by using the discard() method:
fruits.discard("apple")
print(fruits)
# Remove the last item by using the pop() method:
fruits.pop()
print(fruits)
# The clear() method empties the set:
fruits.clear()
print(fruits)
# The del keyword will delete the set completely:
del fruits
# Join two sets:
set1 = {"a", "b", "c"}
set2 = {1, 2, 3}
set3 = set1.union(set2)
print(set3)
# The update() method inserts the items in set2 into set1:

# 4. Dictionary
# A dictionary is a collection which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets, and they have keys and values.
# When to use: Use a dictionary when you have a collection of data that needs to be indexed, and you need a named index. Dictionaries are iterable.
# Example:
# Create a dictionary:
person = {
    "name": "John",
    "age": 36,
    "country": "Norway"
}
print(person)
# Accessing Items:
print(person["name"])
# Change Values:
person["age"] = 40
print(person)
# Loop Through a Dictionary:
for x in person:
    print(x)
    print(person[x])
# Check if Key Exists:
if "age" in person:
    print("Yes, 'age' is one of the keys in the person dictionary")
# Dictionary Length:
print(len(person))
# Adding Items
person["city"] = "Oslo"
print(person)
# Removing Items:
person.pop("age")
print(person)
# The popitem() method removes the last inserted item:
person.popitem()
print(person)
# The del keyword removes the item with the specified key name:
del person["country"]
print(person)
# The clear() method empties the dictionary:
person.clear()
print(person)
# The del keyword can also delete the dictionary completely:
del person
# The dict() Constructor:
person = dict(name="John", age=36, country="Norway")
print(person)

# 5. Stack
# A stack is a collection of objects that are inserted and removed according to the last-in, first-out (LIFO) principle.
# When to use: Use a stack when you need a collection of objects with the last-in, first-out access pattern.
# Example:
# Define a stack:
class Stack:
    """
    Stack follows the Last-In-First-Out (LIFO) principle.
    Operations:
    - Push: O(1) time complexity.
    - Pop: O(1) time complexity.
    - Peek: O(1) time complexity.
    Use Case: Undo mechanisms in text editors.
    """
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item from the stack. Raise an error if the stack is empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item from the stack without removing it. Raise an error if the stack is empty."""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Return True if the stack is empty, False otherwise."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the stack."""
        return str(self.items)
# Create a stack:
stack = Stack()
# Add items to the stack
stack.push("a")
stack.push("b")
stack.push("c")
print("Stack after pushes:", stack)

# Remove items from the stack
stack.pop()
print("Stack after one pop:", stack)

stack.pop()
print("Stack after two pops:", stack)

stack.pop()
print("Stack after three pops:", stack)

# 6. Queue
# A queue is a collection of objects that are inserted and removed according to the first-in, first-out (FIFO) principle.
# When to use: Use a queue when you need a collection of objects with the first-in, first-out access pattern.
# Example:
# Define a queue:
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add an item to the end of the queue."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the item from the beginning of the queue. Raise an error if the queue is empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.popleft()
    
    def peek(self):
        """Return the item at the beginning of the queue without removing it. Raise an error if the queue is empty."""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.items[0]
    
    def is_empty(self):
        """Return True if the queue is empty, False otherwise."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)
    
    def __str__(self):
        """Return a string representation of the queue."""
        return str(list(self.items))
# Create a queue:
queue = Queue()
# Add items to the queue
queue.enqueue("a")
queue.enqueue("b")
queue.enqueue("c")
print("Queue after enqueues:", queue)

# Remove items from the queue
queue.dequeue()
print("Queue after one dequeue:", queue)

queue.dequeue()
print("Queue after two dequeues:", queue)

queue.dequeue()
print("Queue after three dequeues:", queue)

# 7. Linked List
# A linked list is a linear data structure, in which the elements are not stored at contiguous memory locations.
# When to use: Use a linked list when you need a collection of objects with the ability to add and remove items from the beginning, middle, or end of the list.
# The elements in a linked list are linked using pointers.
# Example:
# Create a linked list:
# Define a node:
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
# Define a linked list:
class LinkedList:
    def __init__(self):
        self.head = None
    def print_list(self):
        current_node = self.head
        while current_node:
            print(current_node.data)
            current_node = current_node.next
# Create a linked list:
linked_list = LinkedList()
linked_list.head = Node("a")
second_node = Node("b")
third_node = Node("c")
linked_list.head.next = second_node
second_node.next = third_node
linked_list.print_list()
# remove the second node:
linked_list.head.next = third_node
linked_list.print_list()
# remove the first node:
linked_list.head = second_node
linked_list.print_list()
# add a new node at the beginning:
new_node = Node("d")
new_node.next = linked_list.head
linked_list.head = new_node
linked_list.print_list()
# add a new node at the end:
new_node = Node("e")
current_node = linked_list.head
while current_node.next:
    current_node = current_node.next
current_node.next = new_node
linked_list.print_list()

# 8. Binary Tree
# A binary tree is a tree data structure in which each node has at most two children, which are referred to as the left child and the right child.
# When to use: Use a binary tree when you need a hierarchical collection of objects with the ability to add and remove items from the tree.
# Example:
# Define a Binary Tree:
class BinaryTree:
    """
    Binary Tree is a hierarchical structure with each node having at most two children.
    Operations and their complexities:
    - Insertion: O(n) in the worst case, as we might need to traverse the tree to find the insertion point.
    - Search: O(n) in the worst case, for a skewed tree.
    Use Case: Hierarchical data representation, such as file systems.
    """
    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None
    
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = self.Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = self.Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = self.Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def inorder_traversal(self, node, result=None):
        if result is None:
            result = []
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.value)
            self.inorder_traversal(node.right, result)
        return result

# Create a binary tree:
# Define a node:
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
# Create a binary tree:
root = Node("a")
root.left = Node("b")
root.right = Node("c")
root.left.left = Node("d")
root.left.right = Node("e")
# Print the binary tree:
def print_tree(node):
    if node:
        print_tree(node.left)
        print(node.data)
        print_tree(node.right)
print_tree(root)
# Add a new node:
root.left.left.left = Node("f")
print_tree(root)
# Remove a node:
root.left.left.left = None
print_tree(root)

# 9. Graph
# A graph is a collection of nodes and edges. A graph can be directed or undirected.
# When to use: Use a graph when you need a collection of objects with the ability to connect objects to each other.
# Example:
# Create a graph:
# Define a node:
class Node:
    def __init__(self, data):
        self.data = data
        self.neighbors = []
# Create a graph:
a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")
a.neighbors.append(b)
b.neighbors.append(a)
b.neighbors.append(c)
c.neighbors.append(b)
c.neighbors.append(d)
d.neighbors.append(c)
# Print the graph:
def print_graph(node):
    if node:
        print(node.data)
        for neighbor in node.neighbors:
            print(neighbor.data)
print_graph(a)
# Add a new node:
e = Node("e")
a.neighbors.append(e)
e.neighbors.append(a)
print_graph(a)
# Remove a node:
a.neighbors.remove(b)
b.neighbors.remove(a)
print_graph(a)

# 10. Heap
# A heap is a tree-based data structure which is a complete binary tree and satisfies the heap property.
# When to use: Use a heap when you need a collection of objects with the ability to add and remove items from the heap.
# Example:
# Create a heap:
import heapq
heap = []
# Add items to the heap:
heapq.heappush(heap, 10)
heapq.heappush(heap, 20)
heapq.heappush(heap, 5)
print(heap)
# Remove items from the heap:
heapq.heappop(heap)
print(heap)
# The heap is empty:
heapq.heappop(heap)
print(heap)
# The heap is empty:
heapq.heappop(heap)
print(heap)

# 11. Red-Black Tree
# A red-black tree is a binary search tree which has the following properties:
# 1. Each node is either red or black.
# 2. The root is always black.
# 3. All leaves (NIL) are black.
# 4. If a red node has children, then the children are always black.
# 5. Every path from a node to its descendant NIL nodes has the same number of black nodes.
# When to use: Use a red-black tree when you need a self-balancing binary search tree.
# Example:
# Create a red-black tree:
# Define a node:
class Node:
    def __init__(self, data):
        self.data = data
        self.color = "red"
        self.parent = None
        self.left = None
        self.right = None
# Create a red-black tree:
root = Node(10)
root.color = "black"
root.left = Node(5)
root.left.parent = root
root.left.color = "red"
root.right = Node(15)
root.right.parent = root
root.right.color = "red"
root.left.left = Node(3)
root.left.left.parent = root.left
root.left.left.color = "black"
root.left.right = Node(7)
root.left.right.parent = root.left
root.left.right.color = "black"
root.right.left = Node(13)
root.right.left.parent = root.right
root.right.left.color = "black"
root.right.right = Node(17)
root.right.right.parent = root.right
root.right.right.color = "black"
# Print the red-black tree:
def print_tree(node):
    if node:
        print_tree(node.left)
        print(node.data, node.color)
        print_tree(node.right)
print_tree(root)
# Add a new node:
root.left.left.left = Node(1)
root.left.left.left.parent = root.left.left
root.left.left.left.color = "red"
print_tree(root)
# Remove a node:
root.left.left.left = None
print_tree(root)

# 12. AVL Tree
# An AVL tree is a self-balancing binary search tree. It was the first such data structure to be invented.
# When to use: Use an AVL tree when you need a self-balancing binary search tree.
# Example:
# Create an AVL tree:
# Define a node:
class Node:
    def __init__(self, data):
        self.data = data
        self.height = 1
        self.left = None
        self.right = None
# Create an AVL tree:
def get_height(node):
    if not node:
        return 0
    return node.height

def get_balance(node):
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)

def left_rotate(node):
    right = node.right
    node.right = right.left
    right.left = node
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    right.height = 1 + max(get_height(right.left), get_height(right.right))
    return right

def right_rotate(node):
    left = node.left
    node.left = left.right
    left.right = node
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    left.height = 1 + max(get_height(left.left), get_height(left.right))
    return left

def insert_node(node, data):
    if not node:
        return Node(data)
    if data < node.data:
        node.left = insert_node(node.left, data)
    else:
        node.right = insert_node(node.right, data)
    node.height = 1 + max(get_height(node.left), get_height(node.right))
    balance = get_balance(node)
    if balance > 1:
        if data < node.left.data:
            return right_rotate(node)
        else:
            node.left = left_rotate(node.left)
            return right_rotate(node)
    if balance < -1:
        if data > node.right.data:
            return left_rotate(node)
        else:
            node.right = right_rotate(node.right)
            return left_rotate(node)
    return node

def delete_node(root, data):
    if not root:
        return root
    elif data < root.data:
        root.left = delete_node(root.left, data)
    elif data > root.data:
        root.right = delete_node(root.right, data)
    else:
        if root.left is None:
            temp = root.right
            root = None
            return temp
        elif root.right is None:
            temp = root.left
            root = None
            return temp
        temp = minValueNode(root.right)
        root.data = temp.data
        root.right = delete_node(root.right, temp.data)
    if root is None:
        return root
    root.height = 1 + max(get_height(root.left), get_height(root.right))
    balance = get_balance(root)
    if balance > 1 and get_balance(root.left) >= 0:
        return right_rotate(root)
    if balance < -1 and get_balance(root.right) <= 0:
        return left_rotate(root)
    if balance > 1 and get_balance(root.left) < 0:
        root.left = left_rotate(root.left)
        return right_rotate(root)
    if balance < -1 and get_balance(root.right) > 0:
        root.right = right_rotate(root.right)
        return left_rotate(root)
    return root

def minValueNode(node):
    current = node
    while(current.left is not None):
        current = current.left
    return current

def print_tree(node):
    if node:
        print_tree(node.left)
        print(node.data)
        print_tree(node.right)

root = None
root = insert_node(root, 10)
root = insert_node(root, 20)
root = insert_node(root, 30)
root = insert_node(root, 40)
root = insert_node(root, 50)
root = insert_node(root, 25)
root = delete_node(root, 30)
print_tree(root)

# 13. B-Tree
# A B-tree is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.
# When to use: Use a B-tree when you need a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.
# Example:
# Create a B-tree:
# Define a node:
class Node:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree:
    def __init__(self, t):
        self.root = Node(leaf=True)
        self.t = t  # Minimum degree

    def search(self, k, x=None):
        if isinstance(x, Node):
            i = 0
            while i < len(x.keys) and k > x.keys[i]:
                i += 1
            if i < len(x.keys) and k == x.keys[i]:
                return (x, i)
            elif x.leaf:
                return None
            else:
                return self.search(k, x.children[i])
        else:
            return self.search(k, self.root)

    def split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = Node(leaf=y.leaf)
        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t:(2 * t - 1)]
        y.keys = y.keys[0:(t - 1)]
        if not y.leaf:
            z.children = y.children[t:(2 * t)]
            y.children = y.children[0:t]

    def insert(self, k):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            s = Node(leaf=False)
            self.root = s
            s.children.insert(0, root)
            self.split_child(s, 0)
            self.insert_non_full(s, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(0)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.children[i], k)

# Example usage
btree = BTree(t=2)  # Creates a 2-3-4 Tree
btree.insert(10)
btree.insert(20)
btree.insert(5)
btree.insert(6)
btree.insert(12)
btree.insert(30)
btree.insert(7)
btree.insert(17)

# Searching for some keys
print(btree.search(6))
print(btree.search(15))
print(btree.search(17))

# Print the B-tree
def print_tree(node, level=0):
    print("Level", level, ":", " ".join(str(node.keys)))
    if node.children:
        for child in node.children:
            print_tree(child, level + 1)

print_tree(btree.root)