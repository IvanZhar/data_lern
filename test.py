import timeit


print(timeit.timeit('if 1 > 2:pass', number=10**7))