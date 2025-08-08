# Define a Binary Search Tree:
class BinarySearchTree:
    """
    Binary Search Tree is a hierarchical structure with each node having at most two children, and the left child is less than the parent, and the right child is greater than the parent.
    Operations and their complexities:
    - Insertion: O(h), where h is the height of the tree.
    - Search: O(h), where h is the height of the tree.
    Use Case: Efficient searching and sorting.
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
            