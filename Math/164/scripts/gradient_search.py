import numpy as np

import line_search
import stopping_criteria


def gradient_descent(f, x_0, stopper, step_size=0.01):
	x = x_0

	x_arr = [x]
	scores = [f(x)]
	while True:
		grad = f.grad(x)
		x = x - step_size * grad

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores


def steepest_descent(f, x_0, stopper):
	x = x_0

	x_arr = [x]
	scores = [f(x)]
	while True:
		grad = f.grad(x)

		class searchFunc:
			def __init__(self, f, x, grad):
				self.f = f
				self.x_ = x
				self.grad_ = grad

			def __call__(self, x):
				return f(self.x_ - x * self.grad_)

			def grad(self, x):
				return np.matmul(np.transpose(self.f.grad(self.x_ - x * self.grad_)), self.grad_)

			def second(self, x):
				x_1 = self.x_ - x * self.grad_
				return np.matmul(np.matmul(np.transpose(self.grad_), self.f.second(x_1)), self.grad_)

		g = searchFunc(f, x, grad)
		step_sizes, scores = line_search.newton_1d(g, 1, 
			stopping_criteria.ImprovementStopper(eps=0.0))
		x = x - step_sizes[-1] * grad

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores


def newton_nd(f, x_0, stopper, step_size=1):
	x = x_0

	x_arr = [x_0]
	scores = [f(x)]
	while True:
		grad = f.grad(x)
		second = f.second(x)
		x = x - step_size * np.matmul(np.linalg.inv(second), grad)

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores


def gauss_newton(f, x_0, stopper, step_size=1):
	x = x_0

	x_arr = [x_0]
	scores = [f(x)]
	while True:
		grad = f.grad(x)
		d = np.matmul(np.linalg.inv(np.matmul(np.transpose(grad), grad)), np.transpose(grad))

		x = x - step_size * np.transpose(d) * f(x)

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores


################
### UNTESTED ###
################

def conjugate_gradient(f, x_0, stopper):
	x = x_0
	d_k = -f.grad(x)

	x_arr = [x_0]
	scores = [f(x)]
	while True:
		grad = f.grad(x)
		second = f.second(x)

		alpha_k = -(np.matmul(np.transpose(grad), d_k)) / np.matmul(np.matmul(np.transpose(d_k), second), d_k)
		x = x + alpha_k * d_k

		g_k1 = f.grad(x)
		beta_k = np.matmul(np.matmul(np.transpose(g_k1), second), d_k) / np.matmul(np.matmul(np.transpose(d_k), second), d_k)
		d_k = -g_k1 + beta_k * d_k

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores
