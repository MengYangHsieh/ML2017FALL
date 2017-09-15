import sys
s = list()
s1 = list()
with open(sys.argv[1], 'r', encoding='UTF-8') as f:
	for line in f:
		for word in line.split():
			s1.append(word)
			if word not in s:
				s.append(word)
count = 0
with open('Q1.txt', 'w+', encoding='UTF-8') as fout:
	for word in s:
		fout.write(word + ' ' + str(count) + ' ' + str(s1.count(word)))
		count += 1
		if count != len(s):
			fout.write('\n')
