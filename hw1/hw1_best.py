import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	# 第0列沒有資訊
	if n_row != 0:
		# 每一列只有第3-27格有值(1天內24小時的數值)
		for i in range(3,27):
			if r[i] != "NR" and float(r[i]) != -1:
				data[(n_row-1)%18].append(float(r[i]))
			elif r[i] != "NR" and float(r[i]) == -1:
				# print(r)
				if r[i-1] !="PM2.5":
					r[i] = r[i-1]
				else:
					j = i + 1
					while float(r[j]) == -1:
						j += 1
					r[i] = r[j]
				# print(r)
				data[(n_row-1)%18].append(float(r[i]))
			else:
				data[(n_row-1)%18].append(float(0))
	n_row = n_row+1
text.close()
# for i in range(18):
	# for j in range(10):
		# print(data[i][5759 - j])
	# print("") 

x = []
y = []
data_i = 0
# 每 12 個月
for i in range(11):
	if i == 6:
		data_i += 1
	# 一個月取連續10小時的data可以有471筆
	for j in range(471):
		x.append([])
		# 18種污染物
		for t in range(18):
			if t != 8 and t != 9 and t != 10:
			# if t != 9 and t != 10 and t != 15 and t != 16:
				continue
			# 連續9小時
			for s in range(9):
				x[471*i+j].append(data[t][480*data_i+j+s] )
		y.append(data[9][480*data_i+j+9])
	data_i += 1
x = np.array(x)
y = np.array(y)

# add cubic term
# x = np.concatenate((x,x**3), axis=1)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 1
repeat = 10000

# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
w_best = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
	hypo = np.dot(x,w)
	loss = hypo - y
	cost = np.sum(loss**2) / len(x)
	cost_a  = math.sqrt(cost)
	gra = np.dot(x_t,loss)
	s_gra += gra**2
	ada = np.sqrt(s_gra)
	w = w - l_rate * gra/ada
	print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

w = w_best
# save model
np.save('hw1_best.npy',w)
# read model
w = np.load('hw1_best.npy')
# print(np.sum(w-w_best))

test_x = []
n_row = 0
text = open("test.csv" ,"r")
row = csv.reader(text , delimiter= ",")
count = 0

for r in row:
	need = n_row % 18
	if need != 8 and need != 9 and need != 10:
	# if need != 9 and need != 10 and need != 15 and need != 16:
		if n_row % 18 == 0:
			test_x.append([])
		n_row += 1
		continue
	if n_row %18 == 0:
		test_x.append([])
		for i in range(2,11):
			test_x[n_row//18].append(float(r[i]) )
	else :
		for i in range(2,11):
			if r[i] !="NR" and float(r[i]) != -1:
				test_x[n_row//18].append(float(r[i]))
			elif r[i] != "NR" and float(r[i]) == -1:
				if r[i-1] != "PM2.5":
					r[i] = r[i-1]
				else:
					j = i + 1
					while float(r[j]) == -1:
						j += 1
					r[i] = r[j]
				test_x[n_row//18].append(float(r[i]))
			else:
				# if i == 2:
				# count += 1
				test_x[n_row//18].append(0)
	n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add cubic term
# test_x = np.concatenate((test_x,test_x**3), axis=1)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
	ans.append(["id_"+str(i)])
	a = np.dot(w,test_x[i])
	ans[i].append(a)

# filename = "result/predict.csv"
text = open(sys.argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()