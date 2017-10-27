import pickle
import numpy as np 

Normalizer = 1/0.3
a_s = pickle.load(open('./data/action_state_data','rb'))
as_array = np.zeros((len(a_s),5,3))
for i, ans in enumerate(a_s):
    as_array[i,4,:] = ans['action']
    as_array[i,:4,:] = ans['state'][:,:3]

as_array_0 = np.zeros((len(a_s),5,3))
for i, asn in enumerate(a_s):
    as_array_0[i,4,:] = asn['action']
    as_array_0[i,:4,:] = asn['state'][:,:3] - asn['state'][0,:3]

ends = as_array_0[:,3,:]
print ends.shape

pickle.dump( ends, open( "./data/ends.p", "wb" ) )
