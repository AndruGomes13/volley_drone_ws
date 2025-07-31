import time
import timeit


def f():
    # Simulate some work
    time.sleep(0.1)
    return "done"

N = 20  # Number of times to run the function
res = timeit.repeat(f, repeat=3, number=N)
print(f"Function executed {N} times in: {res} seconds")