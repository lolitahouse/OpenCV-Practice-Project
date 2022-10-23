# -*- coding = utf-8 -*-
# @Time :  10:47
# @Author : lolita
# @File : Logistic_Loss.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

'''
# soup_bowl()
x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.longdouble)
plt_simple_example(x_train, y_train)
plt.close('all')
plt_logistic_squared_error(x_train, y_train)
plt.show()

plt.close('all')
cst = plt_logistic_cost(x_train, y_train)
plt.show()
'''

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

'''
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X_train, y_train, ax)
# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
'''


def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost


'''
# check the cost function
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
'''

x0 = np.arange(0, 6)
x1 = 3 - x0
x1_other = 4 - x0
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# Plot the decision boundary
ax.plot(x0, x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0, x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()


















