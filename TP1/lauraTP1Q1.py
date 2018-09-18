# -*- coding: utf-8 -*-
"""
INF8215 - TP1  - A2018 - Research Methods - Q1 Breadth-First Search 
Created on Tue Sep  4 13:10:12 2018

@author: L.G
"""

import numpy as np
import copy
from queue import Queue
import time 

# Part 1: import montreal weight array
def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()

# Part 2: define the class represeting a solution
class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph 
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.c = idx #c: indicate the index of the current (last added) vertex of the solution
        self.g += self.graph[self.visited[-1],self.c]
        self.visited.append(self.c)
        self.not_visited.remove(self.c)
       

# Part 3: define Breadth-first search (BFS) algorithm that outputs the solution
def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    initial_state = Solution(places, graph)
    frontier = Queue()
    frontier.put(initial_state)
    final_state = False
    
    while not frontier.empty():
        node = frontier.get()
        if not node.not_visited:
            if not final_state:
                best_solution = node
                final_state = True
            else:
                if node.g < best_solution.g:
                    best_solution = node
                    print('best_solution updated         new optimal cost = ',best_solution.g)
        else:
            if len(node.not_visited) == 1:
                    child_node = copy.deepcopy(node)
                    child_node.add(node.not_visited[0])
                    frontier.put(child_node)
            else:
               for unvisited in node.not_visited[:-1]:
                   child_node = copy.deepcopy(node)
                   child_node.add(unvisited)
                   frontier.put(child_node)
    
    return best_solution




#test 1  --------------  OPT. SOL. = 27 time elapsed = 0.04320096969604492 seconds
# solution = [0, 5, 13, 16, 6, 9, 4]
start_time = time.time()
places=[0, 5, 13, 16, 6, 9, 4]
sol = bfs(graph=graph, places=places)
print(sol.g)
print('optimal order =   ', sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

#test 2 -------------- OPT. SOL. = 30 time elapsed = 9.815278053283691 seconds
# solution = [0, 1, 4, 5, 9, 13, 16, 18, 20, 19]
start_time = time.time()
places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
sol = bfs(graph=graph, places=places)
print(sol.g)
print('optimal order =   ', sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

#test 3 -------------- OPT. SOL. = 26 time elapsed = 88.42826247215271 seconds
# solution = [0, 2, 7, 7, 9, 13, 15, 16, 11, 8, 4]
start_time = time.time()
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
sol = bfs(graph=graph, places=places)
print(sol.g)
print('optimal order =   ', sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))




