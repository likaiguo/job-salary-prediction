from __future__ import print_function
import csv
import random

from jobutil import MongoSalaryDB
from jobutil import build_docfreqs
from jobutil import create_reverse_data_index
from jobutil import create_reverse_index
from jobutil import create_sparse_features
from jobutil import select_main_coefficients

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

salary = MongoSalaryDB()

words = [ ]
with open('data/fulldesc/stage2dict.csv') as f:
	dictreader = csv.reader(f)
	for row in dictreader:
		for word in row:
			words.append(word)

def compute_cv_error(reverse_words):
	num_examples = salary.train.count()
	perm = range(num_examples)
	random.shuffle(perm)
	train_offset = (65 * num_examples) / 100
	left_interval = set(range(0, train_offset))
	right_interval = set(range(train_offset, num_examples))

	_, train_reverse_index, train_values = create_reverse_data_index(salary.train, u'SalaryNormalized', left_interval)
	_, cv_reverse_index, cv_values = create_reverse_data_index(salary.train, u'SalaryNormalized', right_interval)

	classifier = LinearRegression()
	train_features = create_sparse_features(salary.train_fulldesc_counter, train_reverse_index, reverse_words, None, left_interval)
	print('Shape %s %s' % (train_features.shape, len(train_values)))
	classifier.fit(train_features, train_values)
	cv_features = create_sparse_features(salary.train_fulldesc_counter, cv_reverse_index, reverse_words, None, right_interval)
	cv_t = classifier.predict(cv_features)
	err = mean_absolute_error(cv_values, cv_t)
	print('mean absolute error', err)
	return err

reverse_words = create_reverse_index(words)

#compute_cv_error(reverse_words)

_, train_reverse_index, train_values = create_reverse_data_index(salary.train, u'SalaryNormalized')
test_forward_index, test_reverse_index, _ = create_reverse_data_index(salary.test)

classifier = LinearRegression()
#classifier = get_pipeline()
X = create_sparse_features(salary.train_fulldesc_counter, train_reverse_index, reverse_words, None)
print('Shape %s %s' % (X.shape, len(train_values)))
classifier.fit(X, train_values)
#classifier.fit(X.toarray(), train_values)
Z = create_sparse_features(salary.test_fulldesc_counter, test_reverse_index, reverse_words, None)
t = classifier.predict(Z)

with open('data/fulldesc/patience.csv', 'w') as f:
	print('Id,SalaryNormalized', file=f)
	for index, value in enumerate(t):
		print('%s,%s' % (test_forward_index[index], value), file=f)
