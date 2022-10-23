# -*- coding = utf-8 -*-
# @Time :  11:40
# @Author : lolita
# @File : Logistic_Regression_Practice.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def sig(z):
    return 1 / (1 + np.exp(-z))


def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j) * (X2 ** j)))
    return np.stack(out, axis=1)


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)


def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code

    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0
        plt.contour(u, v, z, levels=[0.5], colors="g")


# load dataset
X_train, y_train = load_data("data/ex2data1.txt")

'''
# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()
'''


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost(X, y, w, b, lambda_=1):

    m, n = X.shape
    total_cost = 0.0
    for i in range(m):
        f_wb = 0
        for j in range(n):
            f_wb_i = np.dot(X[i, j], w[j])
            f_wb += f_wb_i
        f_wb += b
        f_wb = sigmoid(f_wb)
        total_cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    total_cost = total_cost / m

    return total_cost


'''
m, n = X_train.shape
# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): %.3f' % cost)

# Compute and display cost with non-zero w
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)
print('Cost at test w,b: {:.3f}'.format(cost))
'''


def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    return dj_db, dj_dw


'''
# Compute and display gradient with w initialized to zeroes
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.
dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}')
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}')

# Compute and display cost and gradient with non-zero w
test_w = np.array([0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test_w:', dj_db)
print('dj_dw at test_w:', dj_dw.tolist())
'''


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    # number of training examples
    m = len(X)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing


'''
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
initial_b = -8
# Some gradient descent settings
iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b,
                                   compute_cost, compute_gradient, alpha, iterations, 0)
plot_decision_boundary(w, b, X_train, y_train)                                   
'''


def predict(X, w, b):
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        f_wb = 0
        for j in range(n):
            f_wb_i = np.dot(X[i, j], w[j])
            f_wb += f_wb_i
        f_wb += b
        f_wb = sigmoid(f_wb)
        p[i] = f_wb >= 0.5

    return p


'''
# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
'''

'''
#Compute accuracy on our training set
p = predict(X_train, w, b)
print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))
'''

# load dataset
X_train, y_train = load_data("data/ex2data2.txt")

'''
# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2')
# Set the x-axis label
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

print("Original shape of data:", X_train.shape)
mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)
'''


def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b)
    # You need to calculate this value
    reg_cost = 0.
    for j in range(n):
        reg_cost += (w[j] ** 2)  # scalar
    reg_cost = (lambda_ / (2 * m)) * reg_cost  # scalar
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost


'''
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)
'''


def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for i in range(n):
        dj_dw[i] = dj_dw[i] + (lambda_ / m) * w[i]

    return dj_db, dj_dw


'''
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5

lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )
'''

# Initialize fitting parameters
np.random.seed(1)
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.
# Set regularization parameter lambda_ to 1 (you can try varying this)
lambda_ = 0.01;
# Some gradient descent settings
iterations = 10000
alpha = 0.01

w, b, J_history, _ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

plot_decision_boundary(w, b, X_mapped, y_train)
# Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))










