import math
import random

from pymongo import MongoClient

from scipy.sparse import csr_matrix

from sklearn.linear_model import Lasso, LassoLars

class MongoSalaryDB(object):
	def __init__(self):
		connection = MongoClient()
		self.salary_db = connection['salary']
		self.train = self.salary_db.train
		self.train_fulldesc_counter = self.salary_db.train_fulldesc_counter
		self.train_fulldesc_docfreq = self.salary_db.train_fulldesc_docfreq
		self.test = self.salary_db.test
		self.test_fulldesc_counter = self.salary_db.test_fulldesc_counter
		self.test_fulldesc_docfreq = self.salary_db.test_fulldesc_docfreq

	def __getitem__(self, key):
		return self.salary_db[key]

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

def select_important_words(in_words, salary, domain, field):
	random.shuffle(in_words)
	chunks = split_in_chunks(in_words, 118)

	multi_reverse_words = [ ]
	for chunk in chunks:
		multi_reverse_words.append(create_reverse_index(chunk))

	_, reverse_index, y = create_reverse_data_index(salary[domain], u'SalaryNormalized')

	def salary_collection(name):
		full_name = '_'.join([ domain, field, name ])
		return salary[full_name]

	out_words = [ ]
	for i, reverse_words in enumerate(multi_reverse_words):
		doc_freqs = build_docfreqs(salary_collection('docfreq'), chunks[i])
		X = create_sparse_features(salary_collection('counter'), reverse_index, reverse_words, doc_freqs)
		main_coef = select_main_coefficients(X.toarray(), y, 400.0)
		chunk = chunks[i]
		for index in main_coef:
			out_words.append(chunk[index])
		print('chunk %d: %d words' % (i, len(main_coef)), map(lambda x: chunk[x], main_coef))

	return out_words

def select_main_coefficients(X, values, alpha):
	classifier = Lasso(alpha=alpha)
	classifier.fit(X, values)
	coef = classifier.coef_
	main_coef = [ i for i, value in enumerate(coef) if value != 0 ]
	return main_coef

def split_in_chunks(arr, num_chunks):
	num_elems = len(arr)
	chunk_size, r = divmod(num_elems, num_chunks)

	chunks = [ ]
	left = 0
	for i in xrange(num_chunks):
		stride = chunk_size + (1 if i < r else 0)
		right = left + stride
		chunk = arr[left:right]
		chunks.append(chunk)
		left = right

	return chunks
