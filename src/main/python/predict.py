from __future__ import print_function
import csv

with open('data/pca/microdict.txt') as f:
	dictreader = csv.reader(f)
	for row in dictreader:
		print(len(row))
