import math

from scipy.sparse import csr_matrix

from sklearn.linear_model import Lasso, LassoLars

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

def create_sparse_features(data_coll, reverse_index, word_map, df, indices=None):
	skip_indices = indices is None
	values = [ ]
	row_list = [ ]
	column_list = [ ]
	num_docs = data_coll.count()
	percent = 0
	j = 0
	for i, row in enumerate(data_coll.find()):
		newPercent = (i * 20) / num_docs
		if newPercent != percent:
			print('%d%% done' % (5 * newPercent,))
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
					value = (1 + math.log(counter)) * math.log(num_docs / df[col])
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
