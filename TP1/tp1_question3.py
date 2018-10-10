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
            self.g += self.graph[self.visited[i], self.visited[i+1]]

def shaking(sol, k):
    for i in range(k):
        index1 = random.randrange(1, len(sol.visited[:-1]))
        index2 = random.randrange(1, len(sol.visited[:-1]))
        while index1 == index2:
            index2 = random.randrange(1, len(sol.visited[:-1]))
        sol.swap(index1, index2)
    return sol

def local_search_2opt(sol):
    #init
    current_solution = copy.deepcopy(sol)
    index1 = 1
    found_local_optimum = False
    found_better_cost = False

    while not found_local_optimum:
        #Considérer chaque pair d'indice i, j
        for i in range(index1,len(sol.visited)-3):
            for j in range(i+2, len(sol.visited)-1):
                if not found_better_cost:
                    # Si l’échange donne un plus bas coût, on le réalise
                    sol.swap(i,j)
                    if sol.g < current_solution.g:
                        found_better_cost = True
                        break
                    else:
                        sol.swap(i,j) #swap back.
            if found_better_cost:
                break

        if found_better_cost :
            current_solution = copy.deepcopy(sol)
            found_better_cost = False
        else:
            found_local_optimum = True
            return current_solution

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

def vns(sol, k_max, t_max):
    """
    Performs the VNS algorithm
    """
    timeout = time.time() + t_max
    best_sol = copy.deepcopy(sol)

    while time.time() < timeout:
        current_sol = copy.deepcopy(best_sol)
        current_sol = shaking(current_sol, k_max)
        current_sol = local_search_2opt(current_sol)
        if current_sol.g < best_sol.g:
            best_sol = copy.deepcopy(current_sol)
    return best_sol


####################################################
#               1.2 EXPERIMENTATION                #
####################################################
graph = read_graph()
# test 1  --------------  OPT. SOL. = 27
places=[0, 5, 13, 16, 6, 9, 4]
sol = initial_sol(graph=graph, places=places)
start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

#test 2  --------------  OPT. SOL. = 30
places=[0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
sol = initial_sol(graph=graph, places=places)

start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)

print("--- %s seconds ---" % (time.time() - start_time))

# test 3  --------------  OPT. SOL. = 26
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
sol = initial_sol(graph=graph, places=places)

start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

# test 4  --------------  OPT. SOL. = 40
places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
sol = initial_sol(graph=graph, places=places)

start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))


