SMALL_NUM_TRIALS = 10000
MEDIUM_NUM_TRIALS = 100000
LARGE_NUM_TRIALS = 1000000

def truncate_float(to_truncate, digits):
    nbDecimals = len(str(to_truncate).split('.')[1]) 
    if nbDecimals <= digits:
        return to_truncate
    stepper = 10.0 ** digits
    return math.trunc(stepper * to_truncate) / stepper
