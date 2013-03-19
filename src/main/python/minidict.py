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

t0 = time.time()

salary = MongoSalaryDB()

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

random.shuffle(words)
num_words = len(words)
num_chunks = 118
chunk_size = num_words / num_chunks
r = num_words % num_chunks

chunks = [ ]
left = 0
for i in xrange(num_chunks):
	num_elems = chunk_size + (1 if i < r else 0)
	right = left + num_elems
	chunk = words[left:right]
	chunks.append(chunk)
	left = right

multi_reverse_words = [ ]
for chunk in chunks:
	multi_reverse_words.append(create_reverse_index(chunk))

_, reverse_index, y = create_reverse_data_index(salary.train, u'SalaryNormalized')

main_words = [ ]
for i, reverse_words in enumerate(multi_reverse_words):
	doc_freqs = build_docfreqs(salary.train_fulldesc_docfreq, chunks[i])
	X = create_sparse_features(salary.train_fulldesc_counter, reverse_index, reverse_words, doc_freqs)
	main_coef = select_main_coefficients(X.toarray(), y, 400.0)
	chunk = chunks[i]
	for index in main_coef:
		main_words.append(chunk[index])
	print('chunk %d: %d words' % (i, len(main_coef)), map(lambda x: chunk[x], main_coef))
	dt = time.time() - t0
	print("done in %fm" % (dt / 60))

with open('data/fulldesc/microdict.txt', 'w') as f:
	microdict_writer = csv.writer(f)
	microdict_writer.writerow(main_words)
