import numpy as np
import copy
from queue import Queue
import time
import heapq
import random
from random import shuffle, randint


####################################################
#      DEFINITION DE CLASSES ET DE FONCTIONS       #
####################################################
class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]     est le sommet de départ.
        pm = places[-1]    est le sommet d'arrivé.
        """
        #Init : Créer la solution racine (S_root)
        self.g = 0  # current cost
        self.graph = graph
        self.visited = [places[0]]  # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        current_place = self.visited[-1] #get the last place
        next_place = places[idx]
        self.g += self.graph[current_place,next_place]
        self.visited.append(next_place)
        self.not_visited.remove(next_place)

    def swap(self, index1, index2):
        self.visited[index1], self.visited[index2] = self.visited[index2], self.visited[index1]
        self.g = 0
        for i in range (len(self.visited)-1):
            print(self.g)
            self.g += self.graph[self.visited[i], self.visited[i+1]]

def shaking(sol, k):
    for i in range(k):
        index1 = random.randrange(1, len(sol.visited[:-1]))

        index2 = random.randrange(1, len(sol.visited[:-1]))
        while index1 == index2:
            index2 = random.randrange(1, len(sol.visited[:-1]))
        sol.swap(index1, index2)
        print(sol.visited)
        print(sol.g)
    return sol

def local_search_2opt(sol):
    for i in range()

def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')


def initial_sol(graph, places):
    """
    Return a completed initial solution
    """
    solution = Solution(places, graph)
    return dfs(solution, places)

def dfs(solution, places):
    """
    Performs a Depth-First Search
    """
    while True:
        print("-------")
        print(solution.visited)
        print(solution.g)
        if len(solution.not_visited) == 1:
            solution.add(places.index(solution.not_visited[0]))
            return solution
        elif len(solution.not_visited[:-1]) == 1:
            solution.add(places.index(solution.not_visited[0]))
        else:
            idx_not_visited = random.randrange(1, len(solution.not_visited[:-1]))
            attraction = solution.not_visited[idx_not_visited - 1]
            solution.add(places.index(attraction))
    return None


####################################################
#               1.2 EXPERIMENTATION                #
####################################################
graph = read_graph()

places=[0, 5, 13, 16, 6, 9, 4]
sol = initial_sol(graph=graph, places=places)
shaking(sol, 3)
