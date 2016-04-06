# Enter your code here. Read input from STDIN. Print output to STDOUT

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

tmp = raw_input().split()
r, c, N = int(tmp[0]), int(tmp[1]), int(tmp[2])
tmp = raw_input().split()
R, C = int(tmp[0]), int(tmp[1])

img = []
X = []
Yr = []
Yg = []
Yb = []

for i in range(r):
    data = raw_input()
    data.split('\n')
    #print 'DATA IS', list(data)
    pixels = data.split()
    pixVal = [[int(y) for y in x.split(',')] for x in pixels]
    #print 'PIXEL VALUE IS', pixVal
   
    img.append(pixVal)
    r, g, b = zip(*pixVal)

    Yr.append(list(r))
    Yg.append(list(g))
    Yb.append(list(b))
    fet = [[i*N, j*N] for j in range(c)]
    X.append(fet)


'''
for i in range(r):
    data = raw_input()
    data.split('\n')
    #print 'DATA IS', list(data)
    pixels = data.split()
    pixVal = [[int(y) for y in x.split(',')] for x in pixels]
    #print 'PIXEL VALUE IS', pixVal
   
    img.append(pixVal)
    r, g, b = zip(*pixVal)

    Yr.extend(list(r))
    Yg.extend(list(g))
    Yb.extend(list(b))
    fet = [[i*N, j*N] for j in range(c)]
    X.extend(fet)

Y = [np.array(Yr), np.array(Yb), np.array(Yg)]
X = np.array(X)

clf = []

for i in range(3):
    tmpClf = GradientBoostingRegressor(n_estimators = 19)
    tmpClf.fit(X, Y[i])
    clf.append(tmpClf)

'''

Y = [np.array(Yr), np.array(Yb), np.array(Yg)]
X = np.array(X)

#print X
#print Y
#print len(Y)
#print len(Y[0])
#print Y[0][0]
#print Yg
#print Yb

#print len(Y[0])
#print len(Y[0][0])

def colorAvg(a, b, k):
    li = (a/N)
    lj = (b/N)
    #print li,lj,
    ans = 0
    cnt = 0
    for i in range(2):
        for j in range(2):
            if ((li + i < len(Y[0])) and (lj + j < len(Y[0][0]))):
                ans += Y[k][li + i][lj + j]
                cnt += 1
    return (ans/(cnt*1.0))
                
for i in range(R):
    for j in range(C):
        pix = []
        for t in range(3):
            val = int(round(colorAvg(i, j, t)))
            pix.append(val)
        #if (sum(pix) == 0):
        print str(pix[0])+','+str(pix[1])+','+str(pix[2]),
    print  
