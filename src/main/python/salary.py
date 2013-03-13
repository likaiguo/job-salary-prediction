from __future__ import print_function
import collections
import csv
import math
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

from pymongo import MongoClient

from scipy.sparse import csr_matrix

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA, SparsePCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

connection = MongoClient()
salary_db = connection['salary']
train_fulldesc_docfreq = salary_db.train_fulldesc_docfreq
'''
train_data = [ ]
for i, elem in enumerate(salary_db.train.find()):
	if i % 10000 == 0:
		print(i)
	full_desc = elem['FullDescription']
	train_data.append(full_desc)
'''

table_directory = 'data/table'
files = os.listdir(table_directory)
files.sort()

t0 = time.time()

with open('data/pca/results.txt', 'a') as out_file:
	num_docs = 244768
	for filename in files:
		full_filename = os.path.join(table_directory, filename)
		base = os.path.splitext(filename)[0]
		index = int(base[5:])
		if index <= 2217:
			continue
		values = [ ]
		row_list = [ ]
		column_list = [ ]
		with open(full_filename) as f:
			chunkreader = csv.reader(f)
			header = [ ]
			df = [ ]
			for i, row in enumerate(chunkreader):
				if i == 0:
					header = [ elem for elem in row if len(elem) > 0 ]
					print('length of header', len(header))
					for word in header:
						entry = train_fulldesc_docfreq.find_one({ '_id': word })
						if not isinstance(entry, collections.Iterable):
							print('fuck-up', word)
						df.append(entry['value'])
				else:
					for j in xrange(0, len(row) / 2):
						x = int(row[2 * j])
						y = int(row[2 * j + 1])
						col = x - 101 * index
						y = (1 + math.log(y)) * math.log(num_docs / df[col])
						values.append(y)
						row_list.append(i - 1)
						column_list.append(col)
		print('index', index)
		print('index', index, file=out_file)
		X = csr_matrix((values, (row_list, column_list)), [ num_docs, 101 ])
		Y = X.toarray()
		#pca = PCA(n_components=3)
		pca = SparsePCA(n_components=3)
		pca.fit(X.toarray())
		print(csr_matrix(pca.components_), file=out_file)
		dt = time.time() - t0
		print("done in %fm" % (dt / 60))
		#print(pca.explained_variance_ratio_)
		out_file.flush()
		#break

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time.time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
#X_train = vectorizer.fit_transform(train_data)
print("done in %fs" % (time.time() - t0))
#print("n_samples: %d, n_features: %d" % X_train.shape)
print
