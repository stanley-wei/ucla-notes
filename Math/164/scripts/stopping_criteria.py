import numpy as np

class StepSizeStopper:
	def __init__(self, eps=0.01):
		self.eps = eps

	def stop(self, x, scores):
		if len(x) < 2:
			return True
		dist = np.linalg.norm(x[-1] - x[-2])
		return dist < self.r


class ImprovementStopper:
	def __init__(self, eps=0.01):
		self.eps = eps

	def stop(self, x, scores):
		if len(x) < 2 or np.isnan(scores[-1]):
			return True
		return -(scores[-1] - scores[-2]) < self.eps


class NumIterationsStopper:
	def __init__(self, num_iter):
		self.num_iter = num_iter

	def stop(self, x, scores):
		return len(x) > self.num_iter


def combine(stopper_1, stopper_2):
	class CombinedStopper:
		def __init__(self, stopper_1, stopper_2):
			self.s1 = stopper_1
			self.s2 = stopper_2

		def stop(self, x, scores):
			return (self.s1.stop(x, scores) or self.s2.stop(x, scores))

	stopper = CombinedStopper(stopper_1, stopper_2)
	return stopper
