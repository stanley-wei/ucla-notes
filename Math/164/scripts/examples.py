import numpy as np

from gradient_search import conjugate_gradient, gauss_newton, gradient_descent, newton_nd, steepest_descent
import stopping_criteria

def example_gradientNd():
	class f:
		def __init__(self):
			self.Q = np.array([[1, 0], [0, 2]], dtype=float)
			self.b = np.array([[-1], [-0.5]], dtype=float)

		def __call__(self, x):
			return (0.5 * np.matmul(np.matmul(np.transpose(x), self.Q), x) 
				- np.matmul(np.transpose(self.b), x) + 3)

		def grad(self, x):
			return np.matmul(self.Q, x) - self.b

		def second(self, x):
			return self.Q

	x_arr, scores = steepest_descent(f(), np.array([[0],[0]]), 
		stopping_criteria.NumIterationsStopper(num_iter=2))
	print(x_arr)
	print(scores)

	x_arr, scores = newton_nd(f(), np.array([[0],[0]]), 
		stopping_criteria.NumIterationsStopper(num_iter=10))
	print(x_arr)
	print(scores)

	x_arr, scores = gradient_descent(f(), np.array([[0],[0]]), 
		stopping_criteria.NumIterationsStopper(num_iter=10), step_size=0.05)
	print(x_arr)
	print(scores)
 
 
def example_gaussNewton():
	class f:
		def __init__(self):
			return

		def __call__(self, x):
			return (
					(x[0] + 5) ** 2 
					+ (x[1] + 8) ** 2 
					+ (x[2] + 7) ** 2 
					+ 2 * (x[0]**2) * (x[1]**2) 
					+ 4 * (x[0]**2) * (x[2]**2)
				)[0]

		def grad(self, x):
			return np.array([
					[(2 * (x[0] + 5) + 4 * x[0] * (x[1]**2) + 8 * x[0] * (x[2]**2))[0]],
					[(2 * (x[1] + 8) + 4 * x[1] * (x[0]**2))[0]],
					[(2 * (x[2] + 7) + 8 * x[2] * (x[0]**2))[0]]
				])

		def second(self, x):
			return np.array([
					[(2 + 4 * x[1]**2 + 8 * x[2]**2)[0], (8 * x[0] * x[1])[0], (16 * x[0] * x[2])[0]],
					[(8 * x[1] * x[0])[0], (2 + 4 * x[0]**2)[0], 0],
					[(16 * x[2] * x[0])[0], 0, (2 + 8 * x[0]**2)[0]]
				])

	x_0s = [
			np.array([[1], [1], [1]], dtype=float),
			np.array([[-2.3], [0], [0]], dtype=float),
			np.array([[0], [2], [-12]], dtype=float)
		]
	for x_0 in x_0s:
		x_arr, scores = gauss_newton(f(), x_0,
			stopping_criteria.NumIterationsStopper(num_iter=2))
		print(f"x_0: {x_0}\n")
		print(f"x:\n{"\n".join([str(x) for x in x_arr])}")
		print(f"\nscores:\n{"\n".join([str(score) for score in scores])}")


def example_conjugateGradient():
	class f:
		def __init__(self):
			self.Q = np.array([[5, -3], [-3, 2]])
			self.b = np.array([[0], [1]])
			self.c = -7
			return

		def __call__(self, x):
			return 0.5 * np.matmul(np.matmul(np.transpose(x), self.Q), x) - np.matmul(np.transpose(x), self.b) + self.c

		def grad(self, x):
			return np.matmul(self.Q,  x) - self.b

		def second(self, x):
			return self.Q

		def opt(self):
			x_opt = np.matmul(np.linalg.inv(self.Q), self.b)
			print(f"x_opt: {x_opt}")
			print(f"f(x_opt): {self(x_opt)}")

	x_0 = np.array([[0], [0]])
	x_arr, scores = conjugate_gradient(f(), x_0,
		stopping_criteria.ImprovementStopper())
	print(f"x_0: {x_0}\n")
	print(f"x:\n{"\n".join([str(x) for x in x_arr])}")
	print(f"\nscores:\n{"\n".join([str(score) for score in scores])}")

	f().opt()
 