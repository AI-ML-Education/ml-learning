"""
Pattern 01: Hash Map / Counting

🎯 When to Recognize:
- Problem asks about frequencies ("first unique", "majority", "can construct")
- Need to quickly check membership or last-seen index ("two-sum", "subarray sum")
- One-to-one mapping between two sets ("isomorphic strings")
- Grouping by a signature ("group anagrams")

🧠 Mindset / Template:
1. Identify the "key" to track (element, running sum, sorted string, etc.)
2. Decide if you need:
   - Counter (counts of items)
   - Dict (map value → index/other value)
   - Set (membership only)
3. Iterate once, update structure, check condition.
4. Return result.

⏱️ Complexity:
- Time: O(n) typically
- Space: O(k) for number of unique keys
"""


"""
Problem 1: Two sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""

# %%
# Attempt: 1
def two_sum_v1(nums, target):
    for i, ival in enumerate(nums):
        for j, jval in enumerate(nums):
            if ival + jval == target:
                print([i, j])
                return [i, j]
            
# Brute force: O(n2), we need to reduce the complexity.
# %%
two_sum_v1([2, 7, 11, 15], 26)

# %%
# Attempt: 2
def two_sum_v2(nums, target):
    for first in nums:
        second = target - first
        if second in nums:
            found = [nums.index(first), nums.index(second)]
            print(found)
            return found

# Still O(n2), we further need to reduce the complexity.
# Edge case: 
# nums = [3,3]
# target = 6
# two_sum_v2(nums, target)  # returns [0,0] instead of [0,1]

# %%
two_sum_v2([2, 7, 11, 15], 26)

# %%
# Attempt: 3
def two_sum_v3(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen: # O(1) lookup!
            found = [i, seen[need]]
            print(found)
            return found
        seen[x] = i

# complexity now is: O(n)

# %%
two_sum_v3([2, 7, 11, 15], 26)
two_sum_v3([3, 3], 6)

"""
Problem 2: First Unique Character
Given a string s, find the first non-repeating character in the entire sequence 
and return its index. If it does not exist, return -1.

Example 1:
Input: s = "leetcode"
Output: 0

Example 2:
Input: s = "aabb"
Output: -1
"""

# def first_unique_character(s):
#     count = {}
#     for c in s.split():
#         if c not in count:
#             count[c] = 1
#         else:
#             count[c] =+ 1
#     max(count, key=count.get
# %%
# Attempt: 1
def first_unique_character(s):
    count = {}
    for ch in s:
        if ch in count:
            count[ch] = count[ch] + 1
        else:
            count[ch] = 1
    for i, ch in enumerate(s):
        if count[ch] == 1:
            found = True
            return i
    return -1

# Complexity: O(n) as two loops are parallel and not nested!
        
# %%
#first_unique_character("leetcode")
#first_unique_character("loveleetcode")
#first_unique_character("aabb")

# %%
# Attempt: 2
# Using counter
from collections import Counter
def first_unique_character(s):
    count = Counter(s)              # pass 1: frquency
    for i, ch in enumerate(s):      # pass index with freq 1
        if count[ch] == 1:
            return i
    return -1

# Still O(n) but most common in interviews.
# But the pattern is count once and then scan once.


"""
Problem 3: Valid Anagram
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An anagram is when two words/strings contain exactly the same letters with the same frequencies,
just possibly in a different order. e.g., "listen" and "silent"

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false
"""
# %%
from collections import Counter
def valid_anagram(s, t):
    if len(s) == len(t):                # Fail fast!
        if Counter(s) == Counter(t):    # Counter has O(n) complexity.
            return True
        else:
            return False
    else:
        return False

# Complexity: O(n)

# %%
#valid_anagram("anagram", "nagaram")
#valid_anagram("car", "rat")
valid_anagram("apple", "app")


"""
Problem 3: Group Anagrams
Given an array of strings strs, group the anagrams together.
You can return the answer in any order.

Example 1:
Input: ["eat","tea","tan","ate","nat","bat"]
Output: [["eat","tea","ate"],["tan","nat"],["bat"]]

Example 2:
Input: [""]
Output: [[""]]

Example 3:
Input: ["a"]
Output: [["a"]]
"""
from collections import Counter
def group_anagram(strs):
    group = {}
    for s in strs:
        scount = Counter(s)
        if 
