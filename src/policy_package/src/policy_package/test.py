from collections import deque


d = deque(maxlen=10)

for i in range(14):
    d.appendleft(i)
    
print(d)
