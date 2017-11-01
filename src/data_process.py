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



import numpy as np
class OUProcess(object):
    """Ornstein-Uhlenbeck process"""
    def __init__(self, x_size, mu=0, theta=0.15, sigma=0.3):
        self.x = np.ones(x_size) * mu
        self.x_size = x_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
    def noise(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.x_size)
        self.x = self.x + dx
        return self.x