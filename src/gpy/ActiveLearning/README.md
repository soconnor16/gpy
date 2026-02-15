# ActiveLearning Module

Intelligent data sampling for efficient model training.

## Overview

Pool-Based Active learning reduces the amount of labeled data needed by strategically selecting the most informative points from an unlabeled pool.

### The Active Learning Loop

```
1. Train GP on current labeled set
2. Compute acquisition scores for unlabeled points
3. Select and label highest-scoring point(s)
4. Repeat until stopping criterion met
```

## Usage

### Basic Active Learning

```python
from gpy import ActiveLearner, PeriodicKernel
import numpy as np

# Full dataset 
X_full = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
y_full = np.sin(X_full).ravel()  # Pretend this is expensive to compute

# Create active learner
kernel = PeriodicKernel(length_scale=1.0, period=2*np.pi)
learner = ActiveLearner(
    kernel=kernel,
    x_full=X_full,
    y_full=y_full,
    rmse_threshold=0.05,  # Stop when RMSE < 0.05
    max_points=50,        # Or when 50 points used
    optimize_interval=10  # Re-optimize every 10 iterations
)

# Run active learning
learner.learn(
    learning_strategy="uncertainty",
    batch_size=1,
    final_optimization_method="rmse",
    update=True,          # Print progress
    update_interval=10    # Print progress every 10 iterations
)

# Access trained model
predictions = learner.gp.predict(X_full)
print(f"Points used: {len(learner.y_train)}")
```

## Selection Strategies

### Uncertainty Sampling (`"uncertainty"`)

Selects points where the model is most uncertain (highest predictive variance).

```python
learner.learn(learning_strategy="uncertainty")
```

**Best for:** Exploration, when you want broad coverage of the input space.

### Maximum Absolute Error (`"mae"`)

Selects points where the model makes the largest errors.

```python
learner.learn(learning_strategy="mae")
```

**Best for:** Exploitation, when you want to fix specific problem areas.

### Random (`"random"`)

Baseline strategy with uniform random selection.

```python
learner.learn(learning_strategy="random")
```

**Best for:** Comparison baseline, or when domain knowledge suggests uniform sampling.

## Class Reference

### `ActiveLearner(kernel, x_full, y_full, ...)`

**Parameters:**
- `kernel` (Kernel): Kernel for the internal GP model
- `x_full` (array): Complete pool of input features
- `y_full` (array): Complete pool of target values
- `max_points` (int, optional): Maximum training points to use
- `rmse_threshold` (float): Target RMSE for stopping. Default: `0.5`
- `optimize_interval` (int, optional): Iterations between hyperparameter optimization

**Methods:**
- `learn(learning_strategy, batch_size=1, ...)`: Run the active learning loop
- `select_next_point(selection_function, n_points=1)`: Select next point(s) to add

**Attributes:**
- `gp`: The underlying GaussianProcess model
- `x_train`, `y_train`: Current training data
- `x_full`, `y_full`: Complete data pool
- `remaining_indices`: Indices of unlabeled points

## Stopping Criteria

The learning loop stops when any of these conditions is met:

1. **RMSE threshold reached**: Model achieves target accuracy
2. **Max points reached**: Budget exhausted
3. **Pool exhausted**: All points have been labeled

## Batch Active Learning

Select multiple points per iteration:

```python
learner.learn(
    learning_strategy="uncertainty",
    batch_size=5  # Add 5 points per iteration
)
```

Useful when labeling has high fixed cost but low marginal cost (e.g., batched QM calculations).
