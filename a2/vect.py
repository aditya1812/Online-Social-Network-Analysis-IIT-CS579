from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
import scipy as sp

tokens_list = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
indptr =[0]
indices = []
data = []
voc = {}
'''
for d in tokens_list:
    for term in d:
        index = voc.setdefault(term, len(voc))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))
print(indices)
print(data)
print(indptr)
print('.............................')
'''
comb_list = []
#print(tokens_list[0])
for i in tokens_list:
    for j in i:
        comb_list.append(j)
sort_comb_list = sorted(comb_list)
#print(sort_comb_list)
row = []
tokens1_list = tokens_list
#print('...........')
#print(tokens1_list)
index = 0
for i in sort_comb_list:
    index = 0
    for t in tokens1_list:
        #print(t)
        if i in t:
            #print(index)
            row.append(index)
            t.remove(i)
        index += 1
print('row = ')
print(row)
'''
c = Counter(comb_list)
print(sorted(c.items()))
'''
c = {}

uni_list = []

for i in comb_list:
    if i not in uni_list:
        uni_list.append(i)
uni_list = sorted(uni_list)
#print(uni_list)
index = 0
for i in uni_list:
    c[i] = index
    index += 1
#c = sorted(c.items())
#print(c)
column = []
#print(comb_list)
for i in sort_comb_list:
    #print(i)
    column.append(c[i])
print(column)

data = []
for i in range(len(sort_comb_list)):
    data.append(1)
print(data)

X = csr_matrix((data, (row, column)))
#print(X)
print(type(X))
print(X.toarray())
'''
print("--------------")
X1 = X.getcol(1)
X2 = X.getcol(2)
print(X1.toarray())
print(X2.toarray())
'''
'''
new_data = np.concatenate((X1.data, X2.data))
new_indices = np.concatenate((X1.indices, X2.indices))
new_ind_ptr = X2.indptr + len(X2.data)
new_ind_ptr = new_ind_ptr[1:]
new_ind_ptr = np.concatenate((X1.indptr, new_ind_ptr))
print(csr_matrix((new_data, new_indices, new_ind_ptr)).toarray())
'''
#h = hstack((X1,X2), format = 'csr')
#print(h)

mat = csr_matrix((data, (row, column))).toarray()

#..................Frequency code..................


freq = 1
#print(len(mat))
#print(mat.shape[1])
count = {}
min_freq = 1
for i in range(mat.shape[1]):
    count[i] = 0
    for j in range(mat.shape[0]):
        if mat[j][i] != 0:
            count[i] += 1
    #if count[i] >= min_freq:

print(count)
print(type(mat))

#x = [i for i in count if count[i] >= min_freq)

count = dict((k,v) for k,v in count.items() if v >= min_freq)
count = sorted(count)
print(count)

mat1 = mat[:,count]
print(type(mat1))
mat2 = csr_matrix(mat1)
print(type(mat2))
#print(mat2)
'''
col = 0
for k,v in count.items():
    if v >= min_freq:
        for i in range(mat.shape[0]):
            mat1[i][col] = mat[i][k]
        col += 1

print(mat1.toarray())
'''        
        
test = np.array([[0,1,4,2,1],
        [1,5,4,0,1],
        [4,6,7,9,2],
        [5,8,4,6,9]])
print(type(test))
print(test)
test1 = csr_matrix(test)
print(type(test1))
test2 = test1.toarray()
cos = [0,2,4]
test3 = test2[:,cos]
print(test3)
print(type(test3))
test4 = csr_matrix(test3)
print(type(test4))
print(test4.toarray())


      

    
