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

from parser import read_row_from_file

def build_feature_maps(feature_sources):
	feature_maps = [ ]
	num_features = 0
	for data_coll, filename in feature_sources:
		words = read_row_from_file(filename)
		reverse_words = create_reverse_index(words, num_features)
		feature_maps.append((data_coll, reverse_words))
		num_features += len(words)
	print('num_features', num_features)
	return feature_maps

salary = MongoSalaryDB()

train_feature_sources = [
	(salary['train_fulldesc_counter'], 'data/fulldesc/stage2dict.csv'),
	(salary['train_title_counter'],'data/title/stage3dict.csv'),
	(salary['train_rawloc_counter'], 'data/rawloc/stage2dict.csv'),
]

test_feature_sources = [
	(salary['test_fulldesc_counter'], 'data/fulldesc/stage2dict.csv'),
	(salary['test_title_counter'], 'data/title/stage3dict.csv'),
	(salary['test_rawloc_counter'], 'data/rawloc/stage2dict.csv'),
]

train_feature_maps = build_feature_maps(train_feature_sources)
test_feature_maps = build_feature_maps(test_feature_sources)

def compute_cv_error(reverse_words):
	train = salary['train']
	train_fulldesc_counter = salary['train_fulldesc_counter']
	num_examples = train.count()
	perm = range(num_examples)
	random.shuffle(perm)
	train_offset = (65 * num_examples) / 100
	left_interval = set(range(0, train_offset))
	right_interval = set(range(train_offset, num_examples))

	_, train_reverse_index, train_values = create_reverse_data_index(train, u'SalaryNormalized', left_interval)
	_, cv_reverse_index, cv_values = create_reverse_data_index(train, u'SalaryNormalized', right_interval)

	classifier = LinearRegression()
	train_features = create_sparse_features(train_reverse_index, train_fulldesc_counter, reverse_words, None)
	print('Shape %s %s' % (train_features.shape, len(train_values)))
	classifier.fit(train_features, train_values)
	cv_features = create_sparse_features(cv_reverse_index, train_fulldesc_counter, reverse_words, None)
	cv_t = classifier.predict(cv_features)
	err = mean_absolute_error(cv_values, cv_t)
	print('mean absolute error', err)
	return err

'''
words = read_row_from_file('data/fulldesc/stage2dict.csv')
reverse_words = create_reverse_index(words)
compute_cv_error(reverse_words)
'''

_, train_reverse_index, train_values = create_reverse_data_index(salary['train'], u'SalaryNormalized')
test_forward_index, test_reverse_index, _ = create_reverse_data_index(salary['test'])

classifier = LinearRegression()
#classifier = get_pipeline()
X = create_sparse_features(train_reverse_index, train_feature_maps)
print('Shape %s %s' % (X.shape, len(train_values)))
classifier.fit(X, train_values)
#classifier.fit(X.toarray(), train_values)
Z = create_sparse_features(test_reverse_index, test_feature_maps)
t = classifier.predict(Z)

with open('data/fulldesc/patience.csv', 'w') as f:
	print('Id,SalaryNormalized', file=f)
	for index, value in enumerate(t):
		print('%s,%s' % (test_forward_index[index], value), file=f)
