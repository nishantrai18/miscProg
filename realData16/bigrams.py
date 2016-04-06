import operator

biCont = {}

with open('count_2w.txt', 'r') as f:
	for l in f:
		tmp = l.split('\t')
		a, b = tmp[0], int(tmp[1].strip('\n'))
		a = a.lower()
		biCont[a] = b

freq = sorted(biCont.items(), key=operator.itemgetter(1), reverse = True)

newText = ''

for x in freq[:800]:
	t = x[0]
	if ('<s>' not in t):
		#print t + '|',
		newText += t + '|'

print newText
