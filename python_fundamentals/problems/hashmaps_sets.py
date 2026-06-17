"""
In this section, we will explore the concepts of hashmaps and sets in Python.

Hashmaps, also known as dictionaries in Python, are data structures that store key-value pairs. They are useful for storing and retrieving data efficiently based on unique keys. Unlike arrays, they can store any data type as a key, as long as it is hashable. The space complexity of hashmaps is O(n), where n is the number of key-value pairs in the hashmap. The time complexity for insertion, deletion, and lookup operations is O(1) on average, making hashmaps a powerful tool for many applications. 

Sets, on the other hand, are unordered collections of unique elements. They are useful for performing mathematical set operations such as union, intersection, and difference. Sets in Python are implemented using hash tables, which allows for efficient membership testing and element addition/removal. The space complexity of sets is also O(n), where n is the number of unique elements in the set. The time complexity for adding, removing, and checking membership of elements in a set is O(1) on average.


To practice these concepts, we will implement a few problems that involve hashmaps and sets. These problems will help solidify your understanding of how to use these data structures effectively in Python. We will also reason about why hashmaps and sets are useful in certain scenarios, and how they can be leveraged to solve real-world problems efficiently.
"""

# Two Sum Problem
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.

# %%
def two_sum(nums, target):
    for num1_i, num1 in enumerate(nums):
        num2 = target - num1
        if num2 in nums:
            return num1_i, nums.index(num2)

#two_sum([2, 4, 5, 6, 10, 12], 11)


def two_sum_v2(nums, target):
    seen = []
    for i, num in enumerate(nums):
        comp = target - num
        if seen:
            if comp in seen:
                return i, seen.index(comp)
        seen.append(num)
            
two_sum_v2([2, 4, 5, 6, 10, 12], 11)

# %%

