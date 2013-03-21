from __future__ import print_function

import math

from scipy.sparse import csr_matrix

def create_sparse_features(reverse_index, feature_maps, df=None):
	values = [ ]
	row_list = [ ]
	column_list = [ ]
	for data_coll, word_map in feature_maps:
		print('Extracting features for', data_coll.full_name)
		num_docs = data_coll.count()
		percent = 0
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
	print('#entries', len(values))
	num_features = sum(len(v) for k, v in feature_maps)
	shape = (len(reverse_index), num_features)
	return csr_matrix((values, (row_list, column_list)), shape)
