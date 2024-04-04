import math
import numpy as np
import random
import scipy
from tqdm import tqdm

import utils

#############
# Problem 1 #
#############
def test_birthday_pairs(n, num_trials):
	expected_pairs = (n**2 - n) / 730.0

	sum_pairs = 0
	sum_difference = 0
	for i in tqdm(range(num_trials)):
		birthdays = []
		for j in range(n):
			birthdays.append(random.randint(1, 365))

		num_pairs = 0
		for k in range(n):
			for l in range(k+1, n):
				if birthdays[k] == birthdays[l]:
					num_pairs += 1

		sum_pairs += num_pairs
		sum_difference += (num_pairs - expected_pairs) ** 2

	expected_value = sum_pairs / float(num_trials)
	variance = sum_difference / num_trials

	return expected_value, variance

#############
# Problem 2 #
#############
def test_block_stacking_conditional_failure(p, num_trials):
	total_stacked = {0: 0, 1: 0, 2: 0}

	num_failures = 0
	for i in tqdm(range(num_trials)):
		num_stacked = 0

		rand_nums = scipy.stats.bernoulli.rvs(p, size=3)
		for j in range(3):
			if rand_nums[j]:
				num_stacked += 1
			else:
				total_stacked[num_stacked] += 1
				break
		if num_stacked < 3:
			num_failures += 1

	expected_blocks_stacked = utils.truncate_float((total_stacked[1] + 2 * total_stacked[2]) / float(num_failures), 3)
	for key in total_stacked.keys():
		total_stacked[key] = utils.truncate_float(total_stacked[key] / float(num_failures), 3)

	return total_stacked, expected_blocks_stacked

def num_stacked_first_success(p, num_trials):
	total_stacked = 0

	for i in tqdm(range(num_trials)):
		stacked_count = 0
		while True:
			failed_attempt = False
			rand_nums = scipy.stats.bernoulli.rvs(p, size=3)
			for j in range(3):
				if rand_nums[j]:
					stacked_count += 1
				else:
					failed_attempt = True
					break

			if not failed_attempt:
				break

		total_stacked += stacked_count

	return total_stacked / float(num_trials)


if __name__ == "__main__":
	num_trials = 10000

	print(test_birthday_pairs(50, num_trials))
	print(test_birthday_pairs(100, num_trials))
	print(test_birthday_pairs(250, num_trials))
	print(test_birthday_pairs(500, num_trials))

	# print(test_block_stacking_conditional_failure(0.1, num_trials))
	# print(test_block_stacking_conditional_failure(0.3, num_trials))
	# print(test_block_stacking_conditional_failure(0.6, num_trials))
	# print(test_block_stacking_conditional_failure(0.8, num_trials))

	# print(num_stacked_first_success(0.8, num_trials))
	# print(num_stacked_first_success(0.6, num_trials))
	# print(num_stacked_first_success(0.3, num_trials))
	# print(num_stacked_first_success(0.1, num_trials))
