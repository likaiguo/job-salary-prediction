from __future__ import print_function
import collections
import csv
import numpy as np
import os
import random
import sys
import time
import matplotlib.pyplot as plt

from pymongo import MongoClient

from scipy.sparse import csr_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso, LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from jobutil import MongoSalaryDB
from jobutil import build_docfreqs
from jobutil import create_reverse_data_index
from jobutil import create_reverse_index
from jobutil import create_sparse_features
from jobutil import select_important_words
from jobutil import select_main_coefficients

def get_pipeline():
	steps = [
		("classify", RandomForestRegressor(n_estimators=50,
			verbose=2,
			n_jobs=1,
			min_samples_split=30,
			random_state=3465343))
	]
	return Pipeline(steps)

table_directory = 'data/table'
files = os.listdir(table_directory)
files.sort()
headers = { }

percent = 0
for i, filename in enumerate(files):
	newPercent = (i * 20) / len(files)
	if newPercent != percent:
		print('%d%% files processed' % (5 * newPercent,))
		percent = newPercent
	full_filename = os.path.join(table_directory, filename)
	base = os.path.splitext(filename)[0]
	index = int(base[5:])
	with open(full_filename, 'r') as f:
		chunkreader = csv.reader(f)
		for row in chunkreader:
			headers[index] = row
			break

words = [ ]
with open('data/pca/results.txt', 'r') as f:
	index = -1
	for line in f:
		tokens = map(int, line.split())
		index = tokens[0]
		for position in tokens[1:]:
			word = headers[index][position]
			words.append(word)
print('#words', len(words))

with open('data/pca/minidict.txt', 'w') as f:
	minidict_writer = csv.writer(f)
	minidict_writer.writerow(words)

if __name__ == '__main__':
	salary = MongoSalaryDB()
	out_words = select_important_words(words, salary, 'train', 'fulldesc')

	with open('data/fulldesc/microdict.txt', 'w') as f:
		microdict_writer = csv.writer(f)
		microdict_writer.writerow(out_words)
