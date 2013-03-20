import csv
import os

def parse_headers(directory):
	files = os.listdir(directory)
	files.sort()
	headers = { }

	percent = 0
	for i, filename in enumerate(files):
		newPercent = (i * 20) / len(files)
		if newPercent != percent:
			print('%d%% files processed' % (5 * newPercent,))
			percent = newPercent
		full_filename = os.path.join(directory, filename)
		base = os.path.splitext(filename)[0]
		index = int(base[5:])
		with open(full_filename, 'r') as f:
			chunkreader = csv.reader(f)
			for row in chunkreader:
				headers[index] = row
				break

	return headers

def parse_sparse_results(filename, headers):
	words = [ ]

	with open(filename, 'r') as f:
		index = -1
		for line in f:
			tokens = map(int, line.split())
			index = tokens[0]
			for position in tokens[1:]:
				word = headers[index][position]
				words.append(word)

	return words
