import math
import numpy as np
import random
import scipy
from tqdm import tqdm

#############
# Problem 4 #
#############

def min_dice_roll():
	return min([random.randint(1, 6), random.randint(1, 6), random.randint(1, 6), random.randint(1, 6)])

def test_dice_rolls(num_trials):
	results_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
	for i in tqdm(range(num_trials)):
		results_dict[min_dice_roll()] += 1
	return results_dict

#############
# Problem 6 #
#############

def bernoulli_rth_success(p, r, num_trials):
	expected_value = float(r) / float(p)

	sum_runs = 0
	variance_sum = 0.0
	for i in tqdm(range(num_trials)):
		num_successes = 0
		num_runs = 0
		while num_successes < r:
			trial = scipy.stats.bernoulli.rvs(p, size=1)
			num_runs += 1
			if trial[0]:
				num_successes += 1
		sum_runs += num_runs
		variance_sum += (num_runs - expected_value)**2
	return variance_sum / num_trials, sum_runs / float(num_trials)

if __name__ == "__main__":
	# num_trials = 1000000

	# results = test_dice_rolls(num_trials)
	# print(results)
	# for key, value in results.items():
	# 	print(f"{key}: {round(float(value)/num_trials*100, 3)}%")

	num_trials = 50000

	print(bernoulli_rth_success(0.8, 1, num_trials))
	print(bernoulli_rth_success(0.8, 5, num_trials))
	print(bernoulli_rth_success(0.8, 10, num_trials))
	print(bernoulli_rth_success(0.8, 15, num_trials))
