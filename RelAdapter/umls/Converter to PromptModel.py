
import json
from itertools import chain
import re


test_task = json.load(open('test_tasks.json'))

b = list(test_task.values())
c = list(chain(*b))


test_triplets = []
for line in c:
    test_triplets.append(line)

with open('your_file.txt', 'w') as f:
    for line in test_triplets:
        a = ('    '.join(str(x) for x in line))
        f.write("%s\n" % a)

a = 1