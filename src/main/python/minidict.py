from __future__ import print_function

with open('data/pca/results.txt', 'r') as f:
	num_entries = 0
	index = -1
	for line in f:
		if line.startswith('index'):
			tokens = line.split()
			index = int(tokens[1])
		else:
			tokens = line.strip().split("\t")
			key_tokens = tokens[0][1:-1].split()
			print(index, int(key_tokens[1]))
			num_entries += 1
	print('num_entries', num_entries)
