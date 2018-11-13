# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:03:48 2018

@author: L.G
"""

import numpy as np
import copy
from queue import Queue
import time
import heapq
#import edmonds
import sys

####################################################
#      DEFINITION DE CLASSES ET DE FONCTIONS       #
####################################################

def _getCycle(n, g, visited=None, cycle=None):
    if visited is None:
        visited = set()
    if cycle is None:
        cycle = []
    visited.add(n)
    cycle += [n]
    if n not in g:
        return cycle
    for e in g[n]:
        if e not in visited:
            cycle = _getCycle(e,g,visited,cycle)
    return cycle

def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between
    the current vertex c and the ending vertex pm
    """

    c = sol.visited[-1]
    # pm = sol.not_visited[-1]
    # define variables
    unseenNodes = [0]*(len(sol.not_visited)+1)
    unseenNodes[0] = c
    unseenNodes[1:] = copy.deepcopy(sol.not_visited)
    totalNodes = copy.deepcopy(unseenNodes)
    infinity = 9999999
    shortest_distance = [infinity]*len(unseenNodes)
    #predecessor = [0]*len(unseenNodes)
    #initialize variables
     
    shortest_distance[0] = 0

#visit each node and update weights for al children
    while unseenNodes: # this while visits each node starting from start node
        minNode = None
        for node in unseenNodes: # this for chooses the unvisisted node with the lowest distance
            node_index = totalNodes.index(node)
            if minNode is None:
                minNode = node
                minNode_index = totalNodes.index(minNode)
            elif shortest_distance[node_index] < shortest_distance[minNode_index]:
                minNode = node
                minNode_index = totalNodes.index(minNode)
            allNodes = copy.deepcopy(unseenNodes) #each time we visit a node, its children are all the other unseen nodes
            popidx = allNodes.index(minNode)
            allNodes.pop(popidx)
        for childNode in allNodes:
            child_weight = sol.graph[minNode,childNode]
            childNode_index = totalNodes.index(childNode)
            if (child_weight+ shortest_distance[minNode_index]) < shortest_distance[childNode_index]:
                shortest_distance[childNode_index] = child_weight + shortest_distance[minNode_index]
                # predecessor[childNode_index] = minNode
        popidx = unseenNodes.index(minNode)
        unseenNodes.pop(popidx)
    h = shortest_distance[-1]
    return h


def minimum_spanning_arborescence(sol):
    #step 1: create a dictionary representation of the weighted directed graph
    c = sol.visited[-1]
    unseenNodes = [0]*(len(sol.not_visited)+1)
    unseenNodes[0] = c
    unseenNodes[1:] = copy.deepcopy(sol.not_visited)
    G= {}
    for n in unseenNodes:
        G[n] = {}
        for v in unseenNodes:
            if v == n:
                continue
            else:
                G[n][v] = sol.graph[n,v]
    RG = {}
    for src in G:
        for (dst,c) in G[src].items():
            if dst in RG:
                RG[dst][src] = c
            else:
                RG[dst] = { src : c }
    
    #step 2: create first graph with minumum predecessors for each node
    if c in RG:
        RG[c] = {}
    g = {}
    for n in RG:
        if len(RG[n]) == 0:
            continue
        minimum = sys.maxsize
        s,d = None,None
        for e in RG[n]:
            if RG[n][e] < minimum:
                minimum = RG[n][e]
                s,d = n,e
        if d in g:
            g[d][s] = RG[s][d]
        else:
            g[d] = { s : RG[s][d] }
    #step 3: retreive all cycles from initial graph        
    cycles = []
    visited = set()
    for n in g:
        if n not in visited:
            cycle = _getCycle(n,g,visited)
            cycles.append(cycle)
    rg = {}
    for src in g:
        for (dst,c) in g[src].items():
            if dst in rg:
                rg[dst][src] = c
            else:
                rg[dst] = { src : c }
    #Step 4: colapse all cycles to obtain arborescence graph msa
    for cycle in cycles:
        if c in cycle:
            continue
        allInEdges = []
        minInternal = None
        minInternalWeight = sys.maxsize

        # find minimal internal edge weight
        for n in cycle:
            for e in RG[n]:
                if e in cycle:
                    if minInternal is None or RG[n][e] < minInternalWeight:
                        minInternal = (n,e)
                        minInternalWeight = RG[n][e]
                        continue
                else:
                    allInEdges.append((n,e))        

        # find the incoming edge with minimum modified cost
        minExternal = None
        minModifiedWeight = 0
        for s,t in allInEdges:
            u,v = rg[s].popitem()
            rg[s][u] = v
            w = RG[s][t] - (v - minInternalWeight)
            if minExternal is None or minModifiedWeight > w:
                minExternal = (s,t)
                minModifiedWeight = w
        
        u,w = rg[minExternal[0]].popitem()
        rem = (minExternal[0],u)
        rg[minExternal[0]].clear()
        if minExternal[1] in rg:
            rg[minExternal[1]][minExternal[0]] = w
        else:
            rg[minExternal[1]] = { minExternal[0] : w }
        if rem[1] in g:
            if rem[0] in g[rem[1]]:
                del g[rem[1]][rem[0]]
        if minExternal[1] in g:
            g[minExternal[1]][minExternal[0]] = w
        else:
            g[minExternal[1]] = { minExternal[0] : w }

#step 5: calculate the total cost of the minimum spanning arborescence 
    h = 0
    for node in g:
        for v in g[node]:
            h += g[node][v]
                 
    return  h

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
        #Attributs ajoutés à la question 2.
        self.h = 0

    #Surchargé l'opérateur <
    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        current_place = self.visited[-1] #get the last place
        next_place = places[idx]
        self.g += self.graph[current_place,next_place]
        self.visited.append(next_place)
        self.not_visited.remove(next_place)
        #self.h = fastest_path_estimation(self)
        self.h = minimum_spanning_arborescence(self)


def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')


def A_star(graph, places):
    """
    Performs the A* algorithm
    """
    # 1. blank solution
    root = Solution(graph=graph, places=places)
    # search tree T
    T = []
    heapq.heapify(T)
    heapq.heappush(T, root)
    found = False

    while not found:
        current_sol = heapq.heappop(T)
        #print("-------")
        #print(current_sol.visited)
        #print(current_sol.g + current_sol.h)
        #2. g + fastest_path_estimation(sol)
        for attraction in current_sol.not_visited[:-1]:
            new_sol = copy.deepcopy(current_sol)
            new_sol.add(places.index(attraction))
            heapq.heappush(T, new_sol)
        #S'il reste une seule attraction à visiter.
        if len(current_sol.not_visited) == 1:
            new_sol = copy.deepcopy(current_sol)
            new_sol.add(places.index(current_sol.not_visited[0]))
            heapq.heappush(T, new_sol)
            found = True
            return new_sol
    return None


####################################################
#               1.2 EXPERIMENTATION                #
####################################################
graph = read_graph()

#test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places=[0, 5, 13, 16, 6, 9, 4]
astar_sol = A_star(graph=graph, places=places)
print('test 1 cost: ',astar_sol.g) # result = 27
print(astar_sol.visited)            # result = [0, 5, 13, 16, 6, 9, 4]
print("--- %s seconds ---" % (time.time() - start_time)) 
# resultD= 0.02281665802001953 seconds
# results E = 0.008932828903198242 seconds

#test 2  --------------  OPT. SOL. = 30
start_time = time.time()
places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
astar_sol = A_star(graph=graph, places=places)
print('test2 cost: ',astar_sol.g)       # result = 30
print(astar_sol.visited)                # result = [0, 1, 4, 5, 9, 13, 16, 18, 20, 19]
print("--- %s seconds ---" % (time.time() - start_time)) 
# = 0.22220897674560547 seconds
# results E = 0.0267794132232666 seconds

#test 3  --------------  OPT. SOL. = 26
start_time = time.time()
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
astar_sol = A_star(graph=graph, places=places)
print('test 3 cost: ',astar_sol.g)      # result = 26
print(astar_sol.visited)                # result = [0, 2, 7, 7, 9, 13, 15, 16, 11, 8, 4]
print("--- %s seconds ---" % (time.time() - start_time))
 # = 0.6775338649749756 seconds
 # result E = 0.03422260284423828 seconds

#test 4  --------------  OPT. SOL. = 40
start_time = time.time()
places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
astar_sol = A_star(graph=graph, places=places)
print('test 4 cost: ', astar_sol.g)     # result = 40
print(astar_sol.visited)                # result = [0, 3, 5, 13, 15, 18, 20, 16, 11, 12, 14, 9, 4, 2, 1]
print("--- %s seconds ---" % (time.time() - start_time))
# = 190.72970151901245 seconds
# results E = 1.5668628215789795 seconds