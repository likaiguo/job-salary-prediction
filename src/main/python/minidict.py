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

from jobutil import create_reverse_data_index, create_sparse_features
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

connection = MongoClient()
salary_db = connection['salary']
train = salary_db.train
train_fulldesc_counter = salary_db.train_fulldesc_counter
train_fulldesc_docfreq = salary_db.train_fulldesc_docfreq
test = salary_db.test
test_fulldesc_counter = salary_db.test_fulldesc_counter

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

df = [ ]
for word in words:
	entry = train_fulldesc_docfreq.find_one({ '_id': word })
	df.append(entry['value'])

num_examples = train.count()
perm = range(num_examples)
random.shuffle(perm)
train_offset = (65 * num_examples) / 100
left_interval = set(range(0, train_offset))
right_interval = set(range(train_offset, num_examples))

random.shuffle(words)
num_words = len(words)
num_chunks = 118
chunk_size = num_words / num_chunks
r = num_words % num_chunks

chunks = [ ]
multi_reverse_words = [ ]
left = 0
for i in xrange(num_chunks):
	num_elems = chunk_size + (1 if i < r else 0)
	right = left + num_elems
	chunk = words[left:right]
	chunk_reverse_words = { }
	for i, word in enumerate(chunk):
		chunk_reverse_words[word] = i
	chunks.append(chunk)
	multi_reverse_words.append(chunk_reverse_words)
	left = right

_, all_reverse_index, all_y = create_reverse_data_index(train, u'SalaryNormalized')

main_words = [ ]
for i, reverse_words in enumerate(multi_reverse_words):
	X = create_sparse_features(train_fulldesc_counter, all_reverse_index, reverse_words, df)
	main_coef = select_main_coefficients(X.toarray(), all_y, 400.0)
	chunk = chunks[i]
	for index in main_coef:
		main_words.append(chunk[index])
	print('main chunk(%d::%d) words' % (i, len(main_coef)), map(lambda x: chunk[x], main_coef))
	dt = time.time() - t0
	print("done in %fm" % (dt / 60))

with open('data/pca/microdict.txt', 'w') as f:
	microdict_writer = csv.writer(f)
	microdict_writer.writerow(main_words)
'''
_, train_reverse_index, y = create_reverse_data_index(train, u'SalaryNormalized', left_interval)
_, cv_reverse_index, cv_y = create_reverse_data_index(train, u'SalaryNormalized', right_interval)
test_forward_index, test_reverse_index, _ = create_reverse_data_index(test)

reverse_words = { }
for i, word in enumerate(words):
	reverse_words[word] = i

classifier = LinearRegression()
#classifier = get_pipeline()
X = create_sparse_features(train_fulldesc_counter, train_reverse_index, reverse_words, df, left_interval)
print('X size', X.shape)
print('y size', len(y))
print(classifier.fit(X, y))
#print(classifier.fit(X.toarray(), y))
CV_X = create_sparse_features(train_fulldesc_counter, cv_reverse_index, reverse_words, df, right_interval)
cv_t = classifier.predict(CV_X)
print('error', mean_absolute_error(cv_y, cv_t))
Z = create_sparse_features(test_fulldesc_counter, test_reverse_index, reverse_words, df)
t = classifier.predict(Z)

with open('data/pca/patience.csv', 'w') as f:
	print('Id,SalaryNormalized', file=f)
	for index, value in enumerate(t):
		print('%s,%s' % (test_forward_index[index], value), file=f)
'''
