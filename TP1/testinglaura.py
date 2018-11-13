# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:01:03 2018

@author: L.G
"""
import edmonds
import numpy as np


def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')

graph = read_graph()

places=[0, 5, 13, 16, 6, 9, 4]

g = {}
for n in places:
    g[n] = {}
    for v in places:
        if v == n:
            continue
        else:
            g[n][v] = graph[n,v]
            
root = places[0]

msa = edmonds.mst(root,g)

print(msa)



