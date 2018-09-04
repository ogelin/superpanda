import numpy as np
import copy
from queue import Queue
import time
import heapq

def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')

graph = read_graph()