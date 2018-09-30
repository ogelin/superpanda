# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:36:21 2018

@author: L.G
"""
import numpy as np
import copy
from queue import Queue
import time
import heapq

def fastest_path_estimation(not_visited,graph):
    """
    Returns the time spent on the fastest path between
    the current vertex c and the ending vertex pm
    """
    # c = sol.visited[-1]
    # pm = sol.not_visited[-1]
    # define variables
    
    unseenNodes = copy.deepcopy(not_visited)
    infinity = 9999999
    shortest_distance = [infinity]*len(unseenNodes)
    #predecessor = [0]*len(unseenNodes)
    #initialize variables
  
    shortest_distance[0] = 0
    
    #visit each node and update weights for al children
    while unseenNodes: # this while visits each node starting from start node
        minNode = None
        for node in unseenNodes: # this for chooses the unvisisted node with the lowest distance
            node_index = not_visited.index(node)
            if minNode is None:
                minNode = node
                minNode_index = not_visited.index(minNode)
            elif shortest_distance[node_index] < shortest_distance[minNode_index]:
                minNode = node
                minNode_index = not_visited.index(minNode)
            allNodes = copy.deepcopy(unseenNodes) #each time we visit a node, its children are all the other unseen nodes
            popidx = allNodes.index(minNode)
            allNodes.pop(popidx)
        for childNode in allNodes:
            child_weight = graph[minNode,childNode]
            childNode_index = not_visited.index(childNode)
            if (child_weight+ shortest_distance[minNode_index]) < shortest_distance[childNode_index]:
                shortest_distance[childNode_index] = child_weight + shortest_distance[minNode_index]
               # predecessor[childNode_index] = minNode
        popidx = unseenNodes.index(minNode)
        unseenNodes.pop(popidx)
        
        print('distances=',shortest_distance)
        print('minNode',minNode)
        print('unseen=',unseenNodes)
        
    return shortest_distance[-1] #!!!!!!!! À FAIRE !!!!!!!!!!!!!!!!!!!!!!!!

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')
graph = read_graph()
notvisited=[0, 5, 13, 16, 6, 9, 4]
h = fastest_path_estimation(notvisited,graph)
print(h)