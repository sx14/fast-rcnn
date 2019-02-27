import numpy as np
import random

all_test_passed = True

print('#### Test Example1 ####\n')
num_ver = 3
E_h = np.zeros((num_ver, num_ver))
E_h[0, 1] = 1
E_h[1, 2] = 1
E_e = np.zeros((num_ver, num_ver))

f = [random.normalvariate(0, 1) for i in range(num_ver)]
l = random.randint(0, num_ver-1)
all_test_passed = all_test_passed



def run_test_example(E_h, E_e, f, l):
    """
    :param E_h: hierarchy edge
    :param E_e: exclusion edge
    :param f:   raw scores
    :param l:   label
    """
    print('raw scores:')
    print(f)
    print('label: %d' % l)

    G = hex_setup()
