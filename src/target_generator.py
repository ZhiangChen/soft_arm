import numpy as np
import pickle

ends = pickle.load(open('./data/ends.p', 'rb'))

d_l = 0.01
target = []

for i in range(11):
    target.append(ends + i*d_l*ends)

target = np.array(target).reshape(-1,3)
n_t = target.shape[0]

pickle.dump( target, open( "./data/targets.p", "wb" ) )
t = np.zeros((3,n_t))
t[0] = target[:,0]*10
t[1] = target[:,1]*10
t[2] = (target[:,2]-0.43)*25
print np.max(t[0])
print np.min(t[0])
print np.max(t[1])
print np.min(t[1])
print np.max(t[2])
print np.min(t[2])