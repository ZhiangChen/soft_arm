import numpy as np
import pickle

ends = pickle.load(open('./data/ends.p', 'rb'))

norm = np.linalg.norm(ends, axis=1).reshape(-1,1)
dirction = ends/norm

d_l = 0.003
target = []

for i in range(11):
    target.append(ends + i*d_l*dirction)

target = np.array(target).reshape(-1,3)
n_t = target.shape[0]

pickle.dump( target, open( "./data/targets.p", "wb" ) )
t = np.zeros((3,n_t))
t[0] = target[:,0]*10
t[1] = target[:,1]*10
t[2] = (target[:,2]-0.42)*35
a,b,c,d,e,f = np.max(t[0]), np.min(t[0]), np.max(t[1]), np.min(t[1]), np.max(t[2]), np.min(t[2])

print a
print b
print c
print d
print e
print f
print '\n'
z = [a-b, c-d, e-f]
print np.linalg.norm(z)

