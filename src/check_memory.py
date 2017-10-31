#!/usr/bin/env python

import pickle

M = pickle.load(open('./data/memory.p', 'rb'))
memory = M["memory"]
pointer = M["pointer"]
print pointer
S_DIM = 3
A_DIM = 1

i = 2
while i!='':
    try:
        i = input('i:')
        print memory[i]
        print memory[i,:S_DIM]
        print memory[i,S_DIM:S_DIM+A_DIM]
        print memory[i,-S_DIM-1]
        print memory[i,-S_DIM:]
    except:
        break