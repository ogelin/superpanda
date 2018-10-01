import numpy as np
import copy
from queue import Queue
import time
import heapq


####################################################
#      DEFINITION DE CLASSES ET DE FONCTIONS       #
####################################################
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
    return shortest_distance[-1] #!!!!!!!! À vérifier !!!!!!!!!!!!!!!!!!!!!!!!

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
        self.h = fastest_path_estimation(self)


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
print("--- %s seconds ---" % (time.time() - start_time)) # result = 0.02281665802001953 seconds

#test 2  --------------  OPT. SOL. = 30
start_time = time.time()
places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
astar_sol = A_star(graph=graph, places=places)
print('test2 cost: ',astar_sol.g)       # result = 30
print(astar_sol.visited)                # result = [0, 1, 4, 5, 9, 13, 16, 18, 20, 19]
print("--- %s seconds ---" % (time.time() - start_time)) # = 0.22220897674560547 seconds

#test 3  --------------  OPT. SOL. = 26
start_time = time.time()
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
astar_sol = A_star(graph=graph, places=places)
print('test 3 cost: ',astar_sol.g)      # result = 26
print(astar_sol.visited)                # result = [0, 2, 7, 7, 9, 13, 15, 16, 11, 8, 4]
print("--- %s seconds ---" % (time.time() - start_time)) # = 0.6775338649749756 seconds

#test 4  --------------  OPT. SOL. = 40
start_time = time.time()
places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
astar_sol = A_star(graph=graph, places=places)
print('test 4 cost: ', astar_sol.g)     # result = 40
print(astar_sol.visited)                # result = [0, 3, 5, 13, 15, 18, 20, 16, 11, 12, 14, 9, 4, 2, 1]
print("--- %s seconds ---" % (time.time() - start_time))# = 190.72970151901245 seconds