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


def read_graph():
    return np.loadtxt("contexte/TP1/montreal", dtype='i', delimiter=',')


def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'


    """
    solution = Solution(places, graph)
    frontier = Queue()
    final_solution = None
    frontier.put(solution)
    while frontier.qsize() > 0:
        current_sol = frontier.get()
        print("-------")
        print(current_sol.visited)
        print(current_sol.g)
        if current_sol.visited[-1] == places[-1]:
            if final_solution is None or current_sol.g < final_solution.g:
                final_solution = current_sol
                print("final solution = ", final_solution.g)
        else:
            for attraction in current_sol.not_visited[:-1]:
                new_sol = copy.deepcopy(current_sol)
                new_sol.add(places.index(attraction))
                frontier.put(new_sol)
            if len(current_sol.not_visited) == 1:
                new_sol = copy.deepcopy(current_sol)
                new_sol.add(places.index(current_sol.not_visited[0]))
                frontier.put(new_sol)
    return final_solution

####################################################
#               1.2 EXPERIMENTATION                #
####################################################
graph = read_graph()

#test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places=[0, 2, 7, 13, 11, 4]
sol = bfs(graph=graph, places=places)
print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))