import matplotlib.pyplot as plt
import numpy as np
from gpy.core import ActiveLearner
from gpy.kernels import PeriodicKernel

# generate data to learn on
x = np.linspace(0, 25, 250)
# model a noisy wave with multiple frequencies
y = (
    np.sin(x)
    + 0.5 * np.sin(3 * x)
    + 0.3 * np.cos(5 * x)
    + 0.2 * np.random.default_rng().standard_normal(len(x))
)
# initialize kernel
kernel = PeriodicKernel(4, 1, 2 * np.pi)

# initialize learner
learner = ActiveLearner(
    kernel=kernel,
    x_full=x,  # the full x dataset we want the learner to access
    y_full=y,  # the full y dataset we want the learner to access
    max_points=80,  # (optional) the max points we want the learner to use; here I choose 80% of the points we have
    rmse_threshold=0.3,  # (optional) the rmse value we want the learner to stop at
)

# start the model learning
learner.learn()

# to compare the predictive function to the original data, we will predict on all
# of the datapoints, and plot it with the original data
preds, _ = learner.gp.predict(
    x_test=x,
)

plt.plot(x, y, label="True Function")
plt.plot(x, preds.ravel(), label="Predictive Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("True Function and GPR Predictive Function")
plt.show()
