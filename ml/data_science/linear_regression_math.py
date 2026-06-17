# Understanding Linear regression and optimization mathematically  
# Linear regression is a method to model the relationship between a dependent variable and one or more independent variables.

"""
LINEAR REGRESSION OPTIMIZATION MATHEMATICS
==========================================

This notebook walks through the mathematical foundations of linear regression
and gradient descent optimization step by step.
"""

# =============================================================================
# STEP 1: THE LINEAR RELATIONSHIP
# =============================================================================

"""
Linear regression assumes a linear relationship between dependent and independent variables.

The general equation of a line:
    y = mx + c

Where:
    y  → dependent variable (often a prediction we want to make)
    x  → independent variable / input feature (given data)
    m  → slope of the line (weight coefficient in ML context)
    c  → intercept (bias term in ML context)

GOAL: Find optimal values of m and c that best fit our data points.
      And that would literally be the ML model we are trying to build.
"""

# =============================================================================
# STEP 2: THE OPTIMIZATION PROBLEM
# =============================================================================

"""
When we model data through linear regression, we're trying to fit a line through our data points.
We need to find the optimal values of m and c that represent the "best" line.

But what does "best" mean mathematically?
→ The line that minimizes the difference between predicted and actual values.

This becomes an optimization problem:
    Find m, c that minimize: error between y_predicted and y_actual
"""

# =============================================================================
# STEP 3: MAKING PREDICTIONS
# =============================================================================

"""
Given training data with both x and y values:
    Training set: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}

We start with some arbitrary initial values of m and c (random initialization).

For any input xᵢ, we can calculate the predicted value:
    y_pred_i = m * xᵢ + c

Now we have:
    y_pred_i  → what our model predicts
    y_true_i  → what the actual value is

Question: How do we measure how "wrong" our predictions are?
"""

# =============================================================================
# STEP 4: THE ERROR/LOSS FUNCTION
# =============================================================================

"""
We need a mathematical function to quantify the error between predictions and reality.

For Linear Regression, we use Mean Squared Error (MSE):

    Loss(m, c) = (1/n) * Σ(y_pred_i - y_true_i)²
                         i=1 to n

Substituting y_pred_i = m * xᵢ + c:

    Loss(m, c) = (1/n) * Σ(m * xᵢ + c - y_true_i)²
                         i=1 to n

Why squared error?
    1. Penalizes larger errors more heavily
    2. Always positive (no cancellation of positive/negative errors)
    3. Mathematically convenient (differentiable everywhere)
    4. Creates a convex optimization landscape
"""

# =============================================================================
# STEP 5: THE LEARNING PROBLEM
# =============================================================================

"""
Now we can calculate the error, but we're still not learning anything!

The key insight: We want to MINIMIZE the loss function.

    minimize Loss(m, c) with respect to m and c

or in other words, we want to find the values of m and c that make the loss as small as possible.

This is where GRADIENT DESCENT comes in:
    1. Calculate how the loss changes as we change m and c (gradients)
    2. Move m and c in the direction that reduces the loss
    3. Repeat until we reach the minimum
"""

# =============================================================================
# STEP 6: CALCULATING GRADIENTS (PARTIAL DERIVATIVES)
# =============================================================================

"""
To minimize the loss, we need to know how the loss changes when we change m or c.
This requires calculating partial derivatives (gradients).

Starting with: Loss(m, c) = (1/n) * Σ(m * xᵢ + c - y_true_i)²

Let's define: error_i = m * xᵢ + c - y_true_i

So: Loss(m, c) = (1/n) * Σ(error_i)²

GRADIENT WITH RESPECT TO m (slope):
    ∂Loss/∂m = ∂/∂m [(1/n) * Σ(error_i)²]
             = (1/n) * Σ[2 * error_i * ∂error_i/∂m]
             = (1/n) * Σ[2 * error_i * xᵢ]
             = (2/n) * Σ[(m * xᵢ + c - y_true_i) * xᵢ]

GRADIENT WITH RESPECT TO c (intercept):
    ∂Loss/∂c = ∂/∂c [(1/n) * Σ(error_i)²]
             = (1/n) * Σ[2 * error_i * ∂error_i/∂c]
             = (1/n) * Σ[2 * error_i * 1]
             = (2/n) * Σ[m * xᵢ + c - y_true_i]

Final gradient formulas:
    ∂Loss/∂m = (2/n) * Σ(m * xᵢ + c - y_true_i) * xᵢ
    ∂Loss/∂c = (2/n) * Σ(m * xᵢ + c - y_true_i)

These gradients tell us the rate of change of loss function at a given point m or c.
In other words, these gradients can also be interpreted as the slope of the loss function at that point.
Slope usually tells us not only the rate of change but also direction of change (+ve or -ve).
"""

# =============================================================================
# STEP 7: PARAMETER UPDATE RULES (GRADIENT DESCENT)
# =============================================================================

"""
Now that we have gradients, we can update our parameters to reduce the loss as per the gradient values.

GRADIENT DESCENT UPDATE RULES:
    m_new = m_old - learning_rate * ∂Loss/∂m
    c_new = c_old - learning_rate * ∂Loss/∂c

Where:
    learning_rate (lr) → small positive number (e.g., 0.01)
                        Controls how big steps we take

Why subtract the gradient?
    - Gradient gave us rate and direction of change
    - We want to move in the direction that REDUCES the loss
    - So we want new values of m and c to be in the opposite direction of the gradient
        - If gradient is positive meaning moving on x-axis i.e. m, will increase loss, therefore we subtract to reduce the value of m to decrease the loss
        - If gradient is negative meaning moving on x-axis i.e. m, will decrease loss, therefore we add to increase the value of m to decrease the loss

The learning rate is crucial:
    - If we don't use a learning rate, we might take too large steps and overshoot the minimum.
    - Or in other terms we don't have any control over how big or small the steps we take.
    - If the learning rate is too small, convergence will be very slow.
    - If the learning rate is too large, we might oscillate around the minimum or even diverge.
"""

# =============================================================================
# STEP 8: THE ITERATIVE LEARNING PROCESS
# =============================================================================

"""
GRADIENT DESCENT ALGORITHM:

1. Initialize: m = random_value, c = random_value

2. Repeat until convergence:
   a) Calculate predictions: y_pred_i = m * xᵢ + c for all i

   b) Calculate loss: Loss = (1/n) * Σ(y_pred_i - y_true_i)²

   c) Calculate gradients:
      ∂Loss/∂m = (2/n) * Σ(y_pred_i - y_true_i) * xᵢ
      ∂Loss/∂c = (2/n) * Σ(y_pred_i - y_true_i)

   d) Update parameters:
      m = m - lr * ∂Loss/∂m
      c = c - lr * ∂Loss/∂c

3. Stop when: loss stops decreasing significantly

After convergence: m and c represent the best-fit line!
"""

# =============================================================================
# STEP 9: MATHEMATICAL GUARANTEE - CONVEXITY ANALYSIS
# =============================================================================

"""
How do we know gradient descent will find the global minimum?
Answer: The loss function is CONVEX.

SECOND DERIVATIVE TEST (proving convexity):

For a function to be convex, all second derivatives must be positive.

∂²Loss/∂m² = ∂/∂m [(2/n) * Σ(m * xᵢ + c - y_true_i) * xᵢ]
           = (2/n) * Σ(xᵢ²)
           > 0  (always positive since xᵢ² ≥ 0)

∂²Loss/∂c² = ∂/∂c [(2/n) * Σ(m * xᵢ + c - y_true_i)]
           = (2/n) * n
           = 2 > 0  (always positive)

Since both second derivatives are positive:
    → The loss function is CONVEX
    → Has a unique GLOBAL MINIMUM
    → Gradient descent is GUARANTEED to find it!
"""

# =============================================================================
# STEP 10: CONVERGENCE CONDITIONS
# =============================================================================

"""
How do we know when to stop the gradient descent algorithm?

MATHEMATICAL CONVERGENCE CONDITIONS:

1. Gradient Magnitude: ||∇Loss|| ≈ 0
   When gradients become very small, we're near the minimum.

2. Loss Change: |Loss_t - Loss_{t-1}| < tolerance
   When loss stops decreasing significantly between iterations.

3. Parameter Change: |m_t - m_{t-1}| + |c_t - c_{t-1}| < tolerance
   When parameters stop changing significantly.

At convergence:
    ∂Loss/∂m ≈ 0  and  ∂Loss/∂c ≈ 0

This means we've found the optimal parameters m* and c*!
"""

# =============================================================================
# STEP 11: MATHEMATICAL SUMMARY
# =============================================================================

"""
COMPLETE MATHEMATICAL FRAMEWORK:

PROBLEM: Given data {(xᵢ, yᵢ)}, find optimal line y = mx + c

OBJECTIVE: minimize Loss(m,c) = (1/n) * Σ(mx_i + c - y_i)²

SOLUTION METHOD: Gradient Descent
    1. Compute: ∂Loss/∂m = (2/n) * Σ(mx_i + c - y_i) * x_i
               ∂Loss/∂c = (2/n) * Σ(mx_i + c - y_i)

    2. Update: m ← m - lr * ∂Loss/∂m
               c ← c - lr * ∂Loss/∂c

    3. Repeat until convergence

MATHEMATICAL GUARANTEES:
    ✓ Convex optimization problem (global minimum exists)
    ✓ Gradient descent finds global minimum
    ✓ Solution is unique and optimal

RESULT: Optimal parameters m* and c* that define the best-fit line!
"""

# =============================================================================
# INTUITIVE UNDERSTANDING
# =============================================================================

"""
GEOMETRIC INTERPRETATION:

Think of the loss function as a bowl-shaped surface in 3D:
    - X-axis: parameter m (slope)
    - Y-axis: parameter c (intercept)
    - Z-axis: loss value

Gradient descent is like a ball rolling down this bowl:
    - Always moves downhill (direction of steepest descent)
    - Eventually settles at the bottom (global minimum)
    - The bottom point gives us optimal m* and c*

PHYSICAL ANALOGY:
    Imagine you're blindfolded on a hillside and want to reach the bottom:
    1. Feel the slope around you (compute gradient)
    2. Take a step downhill (parameter update)
    3. Repeat until you reach flat ground (convergence)

This is exactly what gradient descent does mathematically!
"""
