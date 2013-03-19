import math

from pymongo import MongoClient

from scipy.sparse import csr_matrix

from sklearn.linear_model import Lasso, LassoLars

class MongoSalaryDB(object):
	def __init__(self):
		connection = MongoClient()
		salary_db = connection['salary']
		self.train = salary_db.train
		self.train_fulldesc_counter = salary_db.train_fulldesc_counter
		self.train_fulldesc_docfreq = salary_db.train_fulldesc_docfreq
		self.test = salary_db.test
		self.test_fulldesc_counter = salary_db.test_fulldesc_counter
		self.test_fulldesc_docfreq = salary_db.test_fulldesc_docfreq

def build_docfreqs(data_coll, words):
	doc_freqs = [ ]
	for word in words:
		entry = data_coll.find_one({ '_id': word })
		doc_freqs.append(entry['value'])
	return doc_freqs

def create_reverse_data_index(data_coll, key=None, indices=None):
	collect_values = key is not None
	skip_indices = indices is None
	forward_index = [ ]
	original_values = { }
	for i, row in enumerate(data_coll.find()):
		if skip_indices or i in indices:
			row_id = row[u'Id']
			forward_index.append(row_id)
			if collect_values:
				original_values[row_id] = row[key]
	forward_index.sort()

	reverse_index = { }
	values = [ ]
	for i, row_id in enumerate(forward_index):
		reverse_index[row_id] = i
		if collect_values:
			values.append(original_values[row_id])
	return (forward_index, reverse_index, values)

def create_reverse_index(coll):
	reverse_index = { }
	for i, elem in enumerate(coll):
		reverse_index[elem] = i
	return reverse_index

def create_sparse_features(data_coll, reverse_index, word_map, df=None, indices=None):
	skip_indices = indices is None
	values = [ ]
	row_list = [ ]
	column_list = [ ]
	num_docs = data_coll.count()
	percent = 0
	j = 0
	for i, row in enumerate(data_coll.find()):
		newPercent = (i * 10) / num_docs
		if newPercent != percent:
			print('%d%% done' % (10 * newPercent,))
			percent = newPercent
		row_id = row[u'Id']
		if row_id in reverse_index:
			index = reverse_index[row_id]
			arr = row[u'arr']
			for elem in arr:
				word = elem[u'word']
				if word in word_map:
					counter = elem[u'counter']
					col = word_map[word]
					if df is not None:
						value = (1 + math.log(counter)) * math.log(num_docs / df[col])
					else:
						value = counter
					values.append(value)
					row_list.append(index)
					column_list.append(col)
			j += 1
			#if j == 4000:
			#	break
	print('dat row list', len(row_list), len(set(row_list)))
	shape = (len(reverse_index), len(word_map))
	return csr_matrix((values, (row_list, column_list)), shape)

def select_main_coefficients(X, values, alpha):
	classifier = Lasso(alpha=alpha)
	classifier.fit(X, values)
	coef = classifier.coef_
	main_coef = [ i for i, value in enumerate(coef) if value != 0 ]
	return main_coef
