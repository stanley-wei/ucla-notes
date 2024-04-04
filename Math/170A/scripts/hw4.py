import numpy as np
import random
import scipy
from tqdm import tqdm

#############
# Problem 2 #
#############

def test_telescoping_k(max_k, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		j = 1
		while(j < max_k and scipy.stats.bernoulli.rvs((j*(j+2))/((j+1)**2), size=1)):
			j += 1
		if j == max_k:
			counter += 1
	return counter

#############
# Problem 3 #
#############

def test_one_binom(n, p, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		s = np.random.binomial(2*n, p, 1)
		if s == n:
			counter += 1
	return counter

def test_two_binoms(n, p, num_trials):
	counter = 0
	for i in (range(num_trials)):
		s1 = np.random.binomial(2*n, p, 1)
		s2 = np.random.binomial(2*n, p, 1)
		if s1 == s2:
			counter += 1
	return counter

#############
# Problem 4 #
#############

def test_poisson(lam, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		if np.random.poisson(lam=lam, size=1) % 2 == 0:
			counter += 1
	return counter

def test_binomial(n, p, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		if np.random.binomial(n=n, p=p, size=1) % 2 == 0:
			counter += 1
	return counter

def test_geometric(p, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		if np.random.geometric(p=p, size=1) % 2 == 0:
			counter += 1
	return counter

#############
# Problem 5 #
#############

def compare_bernoulli(p, q, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		j = 0
		k = 0
		while not scipy.stats.bernoulli.rvs(p, size=1):
			j += 1
		while not scipy.stats.bernoulli.rvs(q, size=1):
			k += 1
		if j == k:
			counter += 1
	return counter

#############
# Problem 6 #
#############

def infinite_bernoulli_series(p, r, n, num_trials):
	counter = 0
	for i in tqdm(range(num_trials)):
		runs = scipy.stats.bernoulli.rvs(p, size=n)
		if np.count_nonzero(runs) == r and runs[-1]==1:
			counter += 1
	return counter

def main():
	print(infinite_bernoulli_series(0.5, 8, 20, 100000))
	print(infinite_bernoulli_series(0.3, 12, 40, 100000))
	print(infinite_bernoulli_series(0.7, 4, 15, 100000))
	print(infinite_bernoulli_series(0.1, 30, 40, 100000))
	print(infinite_bernoulli_series(0.9, 60, 70, 100000))

main()