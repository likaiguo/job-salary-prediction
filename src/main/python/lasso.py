from __future__ import print_function
import csv
import math
import os
import time

from pymongo import MongoClient

from scipy.sparse import csr_matrix

from jobutil import select_main_coefficients

connection = MongoClient()
salary_db = connection['salary']
train = salary_db.train
train_fulldesc_docfreq = salary_db.train_fulldesc_docfreq

table_directory = 'data/table'
files = os.listdir(table_directory)
files.sort()

t0 = time.time()

train_values = [ ]
for row in train.find():
	train_values.append(row[u'SalaryNormalized'])

with open('data/pca/results.txt', 'a') as out_file:
	num_docs = train.count()
	for filename in files:
		full_filename = os.path.join(table_directory, filename)
		base = os.path.splitext(filename)[0]
		index = int(base[5:])
		values = [ ]
		row_list = [ ]
		column_list = [ ]
		with open(full_filename) as f:
			chunkreader = csv.reader(f)
			header = [ ]
			df = [ ]
			for i, row in enumerate(chunkreader):
				if i == 0:
					header = [ elem for elem in row ]
					for word in header:
						entry = train_fulldesc_docfreq.find_one({ '_id': word })
						df.append(entry['value'])
				else:
					for j in xrange(0, len(row) / 2):
						col = int(row[2 * j])
						y = int(row[2 * j + 1])
						y = (1 + math.log(y)) * math.log(num_docs / df[col])
						values.append(y)
						row_list.append(i - 1)
						column_list.append(col)
		print('index', index, len(header))
		X = csr_matrix((values, (row_list, column_list)),
			[ num_docs, len(header) ])
		main_coef = select_main_coefficients(X.toarray(), train_values, 15.0)
		print(index, ' '.join(map(str, main_coef)), file=out_file)
		dt = time.time() - t0
		print("done in %fm" % (dt / 60))
		out_file.flush()
