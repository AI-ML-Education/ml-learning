"""
Here we will learn more about Python Dictionaries.
"""

# STEP 1: Create a dictionary and access its values
student = {
    "name": "Adam",
    "age": 24,
    "track": "ML"
}

# Accessing values in a dictionary
print("Name:", student["name"])           # direct access (errors if key missing)
print("City:", student.get("city"))       # safe access (returns None if missing)
print("City with default:", student.get("city", "Not set")) # Accessing a non-existent key with get() returns the default value

# Add a new key-value pair
student["city"] = "New York"
print("Updated student dictionary:", student)

# Update an existing key-value pair
student["age"] = 25
print("Updated age:", student)

# Remove a key-value pair
del student["city"]
removed_track = student.pop("track", "Track not found")  # Using pop to remove and get the value
print("Removed track:", removed_track)

# Check membership of a key
print("Is 'name' a key in student?", "name" in student)
print("Is 'city' a key in student?", "city" in student)

## STEP 2: Loops and Comprehensions
scores = {
    "adam": 78,
    "bella": 92,
    "chris": 85,
    "diana": 67
}

# Iterating through keys
print("\nKeys:")
for name in scores.keys():
    print(name)

# Iterating through values
print("\nValues:")
for score in scores.values():
    print(score)

# Iterating through key-value pairs
print("\nKey-Value Pairs:")
for name, score in scores.items():
    print(f"{name}: {score}")

# Dictionary comprehension: keep only scores above 80
top_scorers = {name: score for name, score in scores.items() if score > 80}
print("\nTop Scorers:", top_scorers)

# Dictionary comprehension: curve scores by adding 5 points to each score
curved_scores = {name: score + 5 for name, score in scores.items()}
print("\nCurved Scores:", curved_scores)

# STEP 3: Frequency Maps (Interview Core)
nums = [1, 2, 1, 3, 2, 1, 4, 2, 2]

# Using a dictionary to count frequencies manually
freq_map = {}
for num in nums:
    if num in freq_map:
        freq_map[num] += 1
    else:
        freq_map[num] = 1

print("\nManual frequency map:", freq_map)

# Cleaner manual frequency map using get()
freq_get = {}
for num in nums:
    freq_get[num] = freq_get.get(num, 0) + 1

print("Frequency map using get():", freq_get)

# Using collections.Counter (built-in helper) for frequency counting
from collections import Counter
freq_counter = Counter(nums)
print("Frequency map using Counter:", freq_counter)
print("Most common elements:", freq_counter.most_common(2))  # Get the 2 most common elements

# Real interview mini-problem:
# First non-repeating character in a string
s = "swiss"
char_freq = {}
for ch in s:
    char_freq[ch] = char_freq.get(ch, 0) + 1

first_non_repeating = None
for ch in s:
    if char_freq[ch] == 1:
        first_non_repeating = ch
        break

print("First non-repeating character:", first_non_repeating)