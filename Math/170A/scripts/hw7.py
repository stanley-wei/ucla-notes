import math
import numpy as np
import random
import scipy
from tqdm import tqdm

import utils

#############
# Problem 1 #
#############
def pmf_x_y(num_trials):
	occurrences_dict = {} # keys: X (num_aces) -> Y (num_hearts)
	pmf_dict = {}
	for i in range(0,3):
		occurrences_dict[i] = {}
		pmf_dict[i] = {}
		for j in range(0, 3):
			occurrences_dict[i][j] = 0

	for i in tqdm(range(num_trials)):
		first_draw = random.randint(1, 52)
		second_draw = random.randint(1, 52)
		while second_draw == first_draw:
			second_draw = random.randint(1,52)

		num_aces = 0
		num_hearts = 0
		if first_draw <= 4:
			num_aces += 1
		if second_draw <= 4:
			num_aces += 1

		if first_draw % 4 == 0:
			num_hearts += 1
		if second_draw % 4 == 0:
			num_hearts += 1

		occurrences_dict[num_aces][num_hearts] += 1

	for x_key, x_dict in occurrences_dict.items():
		for y_key, value in x_dict.items():
			pmf_dict[x_key][y_key] = float(value) / num_trials

	return pmf_dict


def test_expected_aces(num_trials): 
	sum_aces = 0
	for i in tqdm(range(num_trials)):
		first_draw = scipy.stats.bernoulli.rvs(4.0/52.0, size=1)
		if first_draw:
			second_draw = scipy.stats.bernoulli.rvs(3.0/51.0, size=1)
			if second_draw:
				sum_aces += 2
			else:
				sum_aces += 1
		else:
			second_draw = scipy.stats.bernoulli.rvs(4.0/51.0, size=1)
			if second_draw:
				sum_aces += 1

	expected_aces = float(sum_aces) / num_trials
	return expected_aces


if __name__ == "__main__":
	print(pmf_x_y(utils.LARGE_NUM_TRIALS))

	# print(test_expected_aces(MEDIUM_NUM_TRIALS))
