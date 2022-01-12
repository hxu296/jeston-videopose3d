import random
import sys
import time

loops = int(sys.argv[1])
print('looping for', loops, 'times')

# increase mem usage
li = []
for _ in range(loops):
  li.append(random.random())

print('li full, sleep for 10 seconds')

time.sleep(10)

print('wake up, cut half mem usage')

li = li[:len(li)//2]

print('mem released, sleep for another 10 seconds')

time.sleep(10)
