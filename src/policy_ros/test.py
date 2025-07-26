
obs = '(1,)'  # Example observation string
elements = obs.strip("()").split(",")
elements = list(filter(None, elements))  # Remove empty strings if any
print(elements)
a = tuple(map(int, elements))
print(a)  # Output: (1,)