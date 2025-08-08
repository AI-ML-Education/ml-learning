# %%
# Binary Tree
# A binary tree is a tree data structure in which each node has at most two children, which are referred to as the left child and the right child.
# When to use: Use a binary tree when you need a hierarchical collection of objects with the ability to add and remove items from the tree.
# Example:
# Define a Binary Tree:
from collections import deque
class BinaryTree:
    """
    Binary Tree is a hierarchical structure with each node having at most two children.
    Enhanced to support various operations:
    - Insertion: O(n) in the worst case for a skewed tree.
    - Search: O(n) in the worst case for a skewed tree.
    - Deletion: O(n) in the worst case for a skewed tree.
    - Traversal: Supports in-order, pre-order, and post-order traversals.
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
        self.root = self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if node is None:
            return self.Node(value)
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        else:
            node.right = self._insert_recursive(node.right, value)
        return node

    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None or node.value == value:
            return node
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)

    def delete(self, value):
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if node is None:
            return node
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            temp_val = self._find_min(node.right).value
            node.value = temp_val
            node.right = self._delete_recursive(node.right, temp_val)
        return node

    def _find_min(self, node):
        while node.left is not None:
            node = node.left
        return node

    def inorder_traversal(self, node, result=None):
        if result is None:
            result = []
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.value)
            self.inorder_traversal(node.right, result)
        return result

    def preorder_traversal(self, node, result=None):
        if result is None:
            result = []
        if node:
            result.append(node.value)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        return result

    def postorder_traversal(self, node, result=None):
        if result is None:
            result = []
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.value)
        return result

    def print_tree(self):
        if not self.root:
            print("The tree is empty")
            return

        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                print(node.value, end=" ")
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            print()  # Newline for a new level

    def print_tree_with_edges(self, node=None, relation="root", depth=0):
        if node is None:
            node = self.root
        if node is not None:
            # Simplify the relation message for debugging
            print(f"{' ' * depth}{relation}: {node.value}")
            self.print_tree_with_edges(node.left, "left", depth + 2)
            self.print_tree_with_edges(node.right, "right", depth + 2)


# %%
# Usage:
# Creating and populating the binary tree
tree = BinaryTree()
tree.insert(5)
tree.insert(3)
tree.insert(7)
tree.insert(1)
tree.insert(4)
tree.insert(6)
tree.insert(8)

# %%
# Traversals
print("In-order Traversal:", tree.inorder_traversal(tree.root))
print("Pre-order Traversal:", tree.preorder_traversal(tree.root))
print("Post-order Traversal:", tree.postorder_traversal(tree.root))

# %%
# Search
print("Searching for 4:", tree.search(4) is not None)

# %%
# Delete
tree.delete(3)
print("In-order Traversal after deleting 3:", tree.inorder_traversal(tree.root))

# %%
# Print the tree
tree.print_tree()

# %%
tree.print_tree_with_edges()
# %%
