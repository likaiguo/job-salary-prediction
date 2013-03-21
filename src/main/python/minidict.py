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
from parser import parse_headers
from parser import parse_sparse_results
from parser import read_row_from_file
from parser import write_row_to_file

def get_pipeline():
	steps = [
		("classify", RandomForestRegressor(n_estimators=50,
			verbose=2,
			n_jobs=1,
			min_samples_split=30,
			random_state=3465343))
	]
	return Pipeline(steps)

def bootstrap_fulldesc(salary):
	headers = parse_headers('data/table')
	words = parse_sparse_results('data/fulldesc/stage0results.csv', headers)
	return words

def bootstrap_title(salary):
	words = [ ]
	docfreq = salary['train_title_docfreq']
	for entry in docfreq.find():
		word = entry['_id']
		words.append(word.encode('utf-8'))
	return words

def bootstrap_rawloc(salary):
	words = [ ]
	docfreq = salary['train_rawloc_docfreq']
	for entry in docfreq.find():
		word = entry['_id']
		words.append(word.encode('utf-8'))
	return words

def iterative_selection(salary, field, stages, bootstrap_function, alphas):
	for i, stage in enumerate(stages):
		if not os.path.isfile(stage):
			if i == 0:
				words = bootstrap_function(salary)
			else:
				prev_words = read_row_from_file(stages[i - 1])
				num_chunks = len(prev_words) // 100
				print('num_chunks:', num_chunks)
				words = select_important_words(prev_words, salary, 'train', field, num_chunks, alphas[i - 1])
			write_row_to_file(stage, words)
		else:
			words = read_row_from_file(stage)
		num_words = len(words)
		print('stage %d: %d words' % (i, num_words))

def main():
	salary = MongoSalaryDB()

	fulldesc_stages = [
		'data/fulldesc/stage1dict.csv',
		'data/fulldesc/stage2dict.csv',
	]
	iterative_selection(salary, 'fulldesc', fulldesc_stages, bootstrap_fulldesc, [ 400.0 ])

	title_stages = [
		'data/title/stage1dict.csv',
		'data/title/stage2dict.csv',
		'data/title/stage3dict.csv',
	]
	iterative_selection(salary, 'title', title_stages, bootstrap_title, [ 10.0, 100.0 ])

	title_stages = [
		'data/rawloc/stage1dict.csv',
		'data/rawloc/stage2dict.csv',
		#'data/rawloc/stage3dict.csv',
	]
	iterative_selection(salary, 'rawloc', title_stages, bootstrap_rawloc, [ 10.0, 100.0 ])

if __name__ == '__main__':
	main()
