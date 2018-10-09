import re

p = 'car.n.01'
n = 't.t.t'

match = re.match(r'(\w+).(\w+).(\d+)', n)
print(match.group())