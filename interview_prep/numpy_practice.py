# %%
import numpy as np

# Create an array from a list
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array 1: {arr1} \nArray 2: {arr2}")

# %%
# Creating arrays with built-in functions
zeros = np.zeros((2, 3))  # 2x3 array of zeros
ones = np.ones((3, 2))   # 3x2 array of ones
full = np.full((2, 2), 7)  # 2x2 array filled with 7
eye = np.eye(3)         # 3x3 identity matrix
empty = np.empty((2, 2))  # 2x2 array with uninitialized values
print(f"Zeros:\n{zeros}\nOnes:\n{ones}\nFull:\n{full}\nIdentity:\n{eye}\nEmpty:\n{empty}")

# %%
# Creating sequences
arange = np.arange(0, 10, 2)  # Array with values from 0 to 10 with step 2 (start, stop, step)
linspace = np.linspace(0, 1, 5)  # Array with 5 values evenly spaced between 0 and 1
print(f"Arange:\n{arange}\nLinspace:\n{linspace}")

# %%
# Creating random arrays
random_uniform = np.random.rand(2, 3)  # 2x3 array of random floats in [0.0, 1.0)
random_normal = np.random.randn(2, 3)  # 2x3 array of random floats from a normal distribution
random_integers = np.random.randint(0, 10, (2, 3))  # 2x3 array of random integers from 0 to 10
print(f"Random Uniform:\n{random_uniform}\nRandom Normal:\n{random_normal}\nRandom Integers:\n{random_integers}")

# %%
# Print the shape and data type of the arrays
print(f"Shape of Array 1: {arr1.shape}, Data Type: {arr1.dtype}")
print(f"Shape of Array 2: {arr2.shape}, Data Type: {arr2.dtype}")

# %%
# Data types
int_array = np.array([1, 2, 3], dtype=np.int32)  # Integer array
float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Float array
bool_array = np.array([True, False, True], dtype=np.bool_)  # Boolean array

# Converting data types
converted_array = int_array.astype(np.float32)  # Convert int array to float

# Array properties
arr = np.array([[1, 2], [3, 4]])
print(f"Array:\n{arr}\nData Type: {arr.dtype}\nSize: {arr.size}\nShape: {arr.shape}\nDimensions: {arr.ndim} \nItem Size: {arr.itemsize} \nTotal Bytes: {arr.nbytes}")

# %%
# Array indexing and slicing
# 1D array indexing
arr1d = np.array([10, 20, 30, 40, 50])
print(arr1d[0])  # First element
print(arr1d[-1])  # Last element
print(arr1d[1:4])  # Slice from index 1 to 3

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[0, 0])  # First element
print(arr2d[1, :])  # Second row
print(arr2d[:, 1])  # Second column
print(arr2d[1:3, 0:2])  # Subarray from row 1 to 2 and column 0 to 1

# Boolean indexing
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 2  # Create a boolean mask
print(arr[mask])  # Apply the mask to get elements greater than 2

# Fancy indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
print(arr[indices])  # Get elements at indices 0, 2, and 4

# %%
# Array reshaping and manipulation
# Reshape an array
arr = np.arange(12)  # Create an array with 12 elements
reshaped = arr.reshape((3, 4))  # Reshape to 3 rows and 4 columns
reshaped_alt = arr.reshape((2, 6))  # Reshape to 2 rows and 6 columns
print(f"Original Array:\n{arr}\nReshaped (3x4):\n{reshaped}\nReshaped (2x6):\n{reshaped_alt}")

# %%
# Flattening an array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
flattened = arr2d.flatten()  # Flatten the 2D array to 1D
raveled = arr2d.ravel()  # Return a flattened view of the array where possible
print(f"Original 2D Array:\n{arr2d}\nFlattened:\n{flattened}\nRaveled:\n{raveled}")

# %%
# Transposing an array
transposed = arr2d.T  # Transpose the 2D array
transposed_alt = np.transpose(arr2d)  # Another way to transpose
print(f"Original 2D Array:\n{arr2d}\nTransposed:\n{transposed}\nTransposed (using np.transpose):\n{transposed_alt}")

# %%
# Adding/Removing dimensions
arr1d = np.array([1, 2, 3])
expanded = np.expand_dims(arr1d, axis=0)  # Add a new dimension
expanded_alt = arr1d[np.newaxis, :]  # Another way to add a new dimension
squeezed = np.squeeze(expanded)  # Remove single-dimensional entries from the shape
print(f"Original 1D Array:\n{arr1d}\nExpanded (new dimension):\n{expanded}\nExpanded (using np.newaxis):\n{expanded_alt}\nSqueezed:\n{squeezed}")

# %%
# Array Operations - Arithmetic
# Element-wise addition
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Basic arithmetic
sum = a + b  # or np.add(a, b) Element-wise addition
diff = a - b  # or np.subtract(a, b) Element-wise subtraction
product = a * b  # or np.multiply(a, b) Element-wise multiplication
divide = a / b  # or np.divide(a, b) Element-wise division
power = a ** 2  # or np.power(a, 2) Element-wise power
print(f"Sum: {sum}\nDifference: {diff}\nProduct: {product}\nDivision: {divide}\nPower: {power}")

# Scalar operations
scalar_add = a + 10  # Add scalar to each element
scalar_multiply = a * 2  # Multiply each element by scalar
print(f"Scalar Addition: {scalar_add}\nScalar Multiplication: {scalar_multiply}")

# Mathematical functions
sqrt = np.sqrt(a)  # Square root of each element
log = np.log(a)  # Natural logarithm of each element
exp = np.exp(a)  # Exponential of each element
sin = np.sin(a)  # Sine of each element
print(f"Square Root: {sqrt}\nLogarithm: {log}\nExponential: {exp}\nSine: {sin}")

# Aggregation functions
arr = np.array([[1, 2, 3], [4, 5, 6]])
sum_all = np.sum(arr)  # Sum of all elements
sum_axis0 = np.sum(arr, axis=0)  # Sum along rows (columns)
sum_axis1 = np.sum(arr, axis=1)  # Sum along columns (rows)
mean = np.mean(arr)  # Mean of all elements
std = np.std(arr)  # Standard deviation of all elements
min_val = np.min(arr)  # Minimum value
max_val = np.max(arr)  # Maximum value
print(f"Sum of all elements: {sum_all}\nSum along axis 0: {sum_axis0}\nSum along axis 1: {sum_axis1}\nMean: {mean}\nStandard Deviation: {std}\nMin: {min_val}\nMax: {max_val}")

# %%
# Array Broadcasting
# Broadcasting allows operations on arrays of different shapes
# Rule: Arrays are compatible when dimensions are equal or one of them is 1

# Scalar and array
arr = np.array([1, 2, 3])
result = arr + 10  # Add scalar to each element or 10 is broadcast to each element
print(f"Array: {arr}\nResult after adding scalar 10: {result}")

# 1D and 2D arrays
a = np.array([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
b = np.array([10, 20, 30])  # shape (3,)
result = a + b  # b is broadcast to each row of a
print(f"Array a:\n{a}\nArray b:\n{b}\nResult after broadcasting:\n{result}")

# Column vector and row vector
col = np.array([[1], [2], [3]])  # shape (3, 1)
row = np.array([10, 20, 30])  # shape (3,)
result = col + row  # row is broadcast to each column of col, shape becomes (3, 3)
print(f"Column Vector:\n{col}\nRow Vector:\n{row}\nResult after broadcasting:\n{result}")

# More complex broadcasting
a = np.ones((3, 4))  # shape (3, 4)
b = np.arange(4)  # shape (4,)
c = np.arange(3)[:, np.newaxis]  # shape (3, 1)
result1 = a + b  # b is broadcast to each row of a, shape becomes (3, 4)
result2 = a + c # c is broadcast to each column of a, shape becomes (3, 4)
print(f"Array a:\n{a}\nArray b:\n{b}\nResult after broadcasting a and b:\n{result1}\nArray c:\n{c}\nResult after broadcasting a and c:\n{result2}")

# %%
# Array Concatenation and Splitting
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenation
concat_1d = np.concatenate((a, b))  # Concatenate 1D arrays
a2d = np.array([[1, 2], [3, 4]])
b2d = np.array([[5, 6], [7, 8]])
concat_axis0 = np.concatenate((a2d, b2d), axis=0)  # Concatenate along rows
concat_axis1 = np.concatenate((a2d, b2d), axis=1)  # Concatenate along columns
print(f"Concatenated 1D Array: {concat_1d}\nConcatenated 2D Array along axis 0:\n{concat_axis0}\nConcatenated 2D Array along axis 1:\n{concat_axis1}")

# %%
# Stack operations
hstack = np.hstack((a2d, b2d))  # Horizontal stack
vstack = np.vstack((a2d, b2d))  # Vertical stack
dstack = np.dstack((a2d, b2d))  # Depth stack
print(f"Horizontal Stack:\n{hstack}\nVertical Stack:\n{vstack}\nDepth Stack:\n{dstack}")

# %%
# Splitting arrays
arr = np.arange(9)
split_equal = np.split(arr, 3)  # Split into 3 equal parts
split_indices = np.split(arr, [2, 5])  # Split at indices 2 and 5
print(f"Original Array: {arr}\nSplit into 3 equal parts: {split_equal}\nSplit at indices 2 and 5: {split_indices}")

# %%
# Linear Algebra Operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
dot_product = np.dot(A, B)  # or A.dot(B) or A @ B
matmul = np.matmul(A, B)  # Matrix multiplication
inner = np.inner(A, B)  # Inner product
outer = np.outer(a, b)  # Outer product of 1D arrays
print(f"Dot Product:\n{dot_product}\nMatrix Multiplication:\n{matmul}\nInner Product:\n{inner}\nOuter Product:\n{outer}")

# matrix operations
det = np.linalg.det(A)  # Determinant of a matrix
inv = np.linalg.inv(A)  # Inverse of a matrix
eigvals, eigvecs = np.linalg.eig(A)  # Eigenvalues and eigenvectors
trace = np.trace(A)  # Trace of a matrix or sum of diagonal elements
print(f"Determinant: {det}\nInverse:\n{inv}\nEigenvalues: {eigvals}\nEigenvectors:\n{eigvecs}\nTrace: {trace}")

# Sorting and Searching
arr = np.array([3,  1, 4, 1, 5, 9, 3, 6])

# Sorting
sorted_arr = np.sort(arr)  # Sort the array
sorted_indices = np.argsort(arr)  # Indices that would sort the array
sorted_2d = np.sort(A, axis=0)  # Sort along specific axis

# Searching
where_greater = np.where(arr > 3)  # Find indices where condition is true
argmax = np.argmax(arr)  # Index of the maximum value
argmin = np.argmin(arr)  # Index of the minimum value
nonzero = np.nonzero(arr)  # Indices of non-zero elements
print(f"Original Array: {arr}\nSorted Array: {sorted_arr}\nSorted Indices: {sorted_indices}\nSorted 2D Array:\n{sorted_2d}\nIndices where elements are greater than 3: {where_greater}\nIndex of maximum value: {argmax}\nIndex of minimum value: {argmin}\nIndices of non-zero elements: {nonzero}")

# %%
# Set Operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])

unique = np.unique(np.array([1, 2, 2, 3, 3, 3, 4]))  # Get unique elements
intersection = np.intersect1d(a, b)  # Elements common to both arrays
union = np.union1d(a, b)  # All unique elements from both arrays
diff = np.setdiff1d(a, b)  # Elements in a but not in b
xor = np.setxor1d(a, b)  # Elements in either a or b but not both or symmetric difference
print(f"Unique Elements: {unique}\nIntersection: {intersection}\nUnion: {union}\nDifference (a - b): {diff}\nSymmetric Difference: {xor}")

# %%
# Memory views and Copies
arr = np.array([1, 2, 3, 4, 5])

# Views vs Copies
view = arr[1:4] # Creates a vuew (shares memory)
copy = arr[1:4].copy()  # Creates a copy (independent memory)
view[0] = 10  # Modifying the view affects the original array
copy[0] = 20  # Modifying the copy does not affect the original array

# Check if arrays share memory
print(f"View shares memory: {np.shares_memory(arr, view)}")
print(f"Copy shares memory: {np.shares_memory(arr, copy)}")

# %%
# Advanced Indexing with meshgrid
x = np.arange(3)
y = np.arange(4)
X, Y = np.meshgrid(x, y) # Create coordinate matrices
print(f"X:\n{X}\nY:\n{Y}")

# %%
# Vectorization and Performance
# Using vectorized operations instead of loops for performance
# Bad practice: using loops (slow)
def slow_operation(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2 + 2 * arr[i] + 1
    return result

# Good practice: using vectorized operations (fast)
def fast_operation(arr):
    return arr ** 2 + 2 * arr + 1

# Using np.vectorize for custom functions
def custom_function(x):
    return x ** 2 if x > 0 else 0

vectorized_function = np.vectorize(custom_function)
result = vectorized_function(np.array([-1, 0, 1, 2, 3]))
print(f"Result of custom vectorized function: {result}")

# %%
# Structured Arrays (for handling heterogeneous data)
# Define a structured array with different data types
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
people = np.array([('Alice', 30, 55.5), ('Bob', 25, 70.0)], dtype=dt)
print(f"Names: {people['name']})")
print(f"Ages: {people['age']})")


# %%
# Common Interview Questions Examples

# 1. Find indices of N largest values
arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
n_largest_indices = np.argpartition(arr, -3)[-3:]  # Get indices of the 3 largest values
n_largest_values = arr[n_largest_indices]
print(f"Indices of 3 largest values: {n_largest_indices}\nValues: {n_largest_values}")

# 2. Moving Average
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
window_size = 3
ma = moving_average(arr, window_size)
print(f"Original Array: {arr}\nMoving Average (window size {window_size}): {ma}")

# 3. Remove duplicates while preserving order
def remove_duplicates(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]
arr = np.array([1, 2, 3, 2, 1, 4, 5])
unique_arr = remove_duplicates(arr)
print(f"Original Array: {arr}\nArray with duplicates removed: {unique_arr}")

# 4. Find Local maxima
def find_local_maxima(arr):
    return np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1  # +1 to adjust for slicing
arr = np.array([1, 3, 2, 5, 4, 6, 5, 7, 8, 7])
local_maxima_indices = find_local_maxima(arr)
local_maxima_values = arr[local_maxima_indices]
print(f"Original Array: {arr}\nLocal maxima indices: {local_maxima_indices}\nLocal maxima values: {local_maxima_values}")


# %%
