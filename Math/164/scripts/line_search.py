import numpy as np


def bisection_search(f, a_0, b_0, eps=0.01, iter=None):
	a = a_0
	b = b_0
	dist = np.sqrt(np.sum(np.power(a-b, 2)))

	i = 0
	while dist > eps or (iter is not None and i<iter):
		med = 0.5 * np.sum((a, b), axis=0)

		grad_med = np.matmul(np.transpose(b-a)/dist, f.grad(med))
		if grad_med > 0:
			b = med
		else:
			a = med

		dist = np.sqrt(np.sum(np.power(a-b, 2)))
		i += 1

	return a, b


def newton_1d(f, x_0, stopper, step_size=0.01):
	x = x_0

	x_arr = [x_0]
	scores = [f(x)]
	while True:
		grad = f.grad(x)
		second = f.second(x)
		x = x - step_size * grad / second

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores


def secant_search(f, x_0, x_1, stopper, step_size=1):
	x_prev = x_0
	x = x_1
	min_f = f(x)

	x_arr = [x_prev, x]
	scores = [f(x_prev), f(x)]
	while True:
		grad = f.grad(x)
		second = (f(x) - f(x_prev)) / (x - x_prev)

		x_prev = x
		x = x - step_size * grad / second

		x_arr.append(x)
		scores.append(f(x))
		if stopper.stop(x_arr, scores):
			break

	return x_arr, scores
