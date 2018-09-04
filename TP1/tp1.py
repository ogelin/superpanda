import numpy as np
import copy
from queue import Queue
import time
import heapq


####################################################
#      DEFINITION DE CLASSES ET DE FONCTIONS       #
####################################################
class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0  # current cost
        self.graph = graph
        self.visited = [places[0]]  # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """


def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')


def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """

####################################################
#               1.2 EXPERIMENTATION                #
####################################################
graph = read_graph()