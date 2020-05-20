"""Example of a solver using numpy tensors.
The problem solves a basic coin change problem, where a given sum must
be achieved as the sum of coin values, minimizing the number of coins.
This is a purely integer problem well suited to showcase numpy matrices.
"""

from mip import Model, MINIMIZE, INTEGER
import numpy as np

model = Model(sense=MINIMIZE)

# we have coins for 1 cent, 2 cents, 5 cents, 10 cents, 20 cents, 50 cents, 1 euro, 2 euros
vals = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1, 2], dtype=float)

# we have a limited amount of coins for each type
available = np.array([5, 5, 5, 5, 5, 5, 2, 0], dtype=int)

# 8 types of coins in total
x = model.add_var_tensor(shape=vals.shape, name="x", var_type=INTEGER)

# objective: minimize number of coins
model.objective = x.sum()

# boundary: amount must be equal to required change, within rouding errors
required_change = 3.74
eps = 0.005

# total value of the coins
amount = x.dot(vals)
print("Value of the coins: %s" % amount)

# these are 2 separate scalar constraints computed with tensor notation
model += (required_change - eps) <= amount
model += amount <= (required_change + eps)

# coins availability
# these are 8 different scalar constraints expressed with tensor notation
model += x <= available, "availability"

# go and see how the constraint lable was expanded
model.write("numpy_tensor_example.lp")

model.optimize()

x_val = np.vectorize(lambda var: var.x)(x)
print("Solution vector: %s" % x_val)

print("Coins:")
for coin_value, pieces in zip(vals, x_val):
    print("%0.2f euro coin: %d" % (coin_value, pieces))
