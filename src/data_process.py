import pickle
import numpy as np 

Normalizer = 1/0.3
aoe = pickle.load(open('action_origin_end.p','rb'))
r_data = aoe['random']
rs_data = aoe['random_step']

cut_r_data = []
cut_rs_data = []
for d in r_data:
    if all(d[0,:]<26):
        cut_r_data.append(d)

for d in rs_data:
    if all(d[0, :] < 26):
        cut_rs_data.append(d)

r_data = np.array(cut_r_data)
rs_data = np.array(cut_rs_data)


r_origins = r_data[:,1,:]
r_ends = r_data[:,2,:]
rs_origins = rs_data[:,1,:]
rs_ends = rs_data[:,2,:]
r_origin = np.average(r_origins,axis=0)
rs_origin = np.average(rs_origins,axis=0)
all_origins = np.concatenate((r_origins,rs_origins))
origin = np.average(all_origins,axis=0)

r_ends_0 = r_ends - r_origin
rs_ends_0 = rs_ends - rs_origin
ends = {'r_ends':r_ends_0, 'rs_ends': rs_ends_0}
with open("ends.p", 'wb') as wfp:
    pickle.dump(ends, wfp)
print r_ends_0.shape
print rs_ends_0.shape
print origin
print '*'*40

new_r_origins = r_origins - r_origin
new_rs_origins = rs_origins - rs_origin

new_r_ends = r_ends - r_origin
new_rs_ends = rs_ends - rs_origin

print r_origin
print rs_origin
print '*'*40
print np.max(new_r_ends,axis=0)*Normalizer
print np.max(new_rs_ends,axis=0)*Normalizer
print np.min(new_r_ends,axis=0)*Normalizer
print np.min(new_rs_ends,axis=0)*Normalizer

r_space = np.max(new_r_ends,axis=0) - np.min(new_r_ends,axis=0)
rs_space = np.max(new_rs_ends,axis=0) - np.min(new_rs_ends,axis=0)

print '*'*40
print r_space 
print rs_space

offset = np.array([.0,.0,-0.35])

print '*'*40
new_r_ends = new_r_ends + offset
new_rs_ends = new_rs_ends + offset
print np.max(new_r_ends,axis=0)
print np.max(new_rs_ends,axis=0)
print np.min(new_r_ends,axis=0)
print np.min(new_rs_ends,axis=0)

print '*'*40
print 'Normalized results:'


new_r_ends = Normalizer * new_r_ends
new_rs_ends = Normalizer * new_rs_ends
print np.max(new_r_ends,axis=0)
print np.max(new_rs_ends,axis=0)
print np.min(new_r_ends,axis=0)
print np.min(new_rs_ends,axis=0)
