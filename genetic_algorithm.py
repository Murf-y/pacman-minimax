# -*- coding: utf-8 -*-
"""GA code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CIEAP63w7YABjU0UbX5Wu3DGOEeZ7NIX

# Fitness function
"""

import os
import sys
import numpy as np
from statistics import mode, mean, median, stdev, variance, quantiles
import math
from tqdm import tqdm
import itertools
from functools import reduce
import operator
import pacman
import layout
import pacman
import multiAgents
import ghostAgents
import textDisplay
from game import Directions
import util

cache = {}
def foodHeuristic(position, foodGrid):
    "*** YOUR CODE HERE ***"

    food_list = foodGrid.asList()

    if len(food_list) == 0:
        return 0
    if len(food_list) == 1:
        return heuristic_found_by_ga_for_food_problem(position, food_list[0])
    
    closest_point = food_list[0]
    furthest_point = food_list[0]

    """
    How it works:
    1. Find the closest point using heuristic found by GA
    2. Find the furthest point using heuristic found by GA
    3. use caching to store previous results (use problem.heuristicInfo to store the results)
    4. return the sum of the estimate DISTANCE from pacman to the closest point and the estimate cost (using heuristic found by GA) from the closest point to the furthest point
    """

    for food in food_list:
        estimated_distance_to_closest = 0
        if str((position, closest_point)) in cache:
            estimated_distance_to_closest = cache[str((position, closest_point))]
        else:
            estimated_distance_to_closest = heuristic_found_by_ga_for_food_problem(position, closest_point)
            cache[str((position, closest_point))] = estimated_distance_to_closest
        
        estimated_distance_to_speculated_closest = heuristic_found_by_ga_for_food_problem(position, food)
        if estimated_distance_to_speculated_closest < estimated_distance_to_closest:
            closest_point = food
            cache[str((position, closest_point))] = estimated_distance_to_speculated_closest
        
        estimated_distance_to_furthest = 0
        if str((position, furthest_point)) in cache:
            estimated_distance_to_furthest = cache[str((position, furthest_point))]
        else:
            estimated_distance_to_furthest = heuristic_found_by_ga_for_food_problem(position, furthest_point)
            cache[str((position, furthest_point))] = estimated_distance_to_furthest
        
        estimated_distance_to_speculated_furthest = heuristic_found_by_ga_for_food_problem(position, food)
        if estimated_distance_to_speculated_furthest > estimated_distance_to_furthest:
            furthest_point = food
            cache[str((position, furthest_point))] = estimated_distance_to_speculated_furthest
    
    return heuristic_found_by_ga_for_food_problem(position, closest_point) + heuristic_found_by_ga_for_food_problem(closest_point, furthest_point)

def max_heuristic(current, goal):
    return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))

def min_heuristic(current, goal):
    return min(abs(current[0] - goal[0]), abs(current[1] - goal[1]))

def diagonal_distance(current, goal):
    dx = abs(current[0] - goal[0])
    dy = abs(current[1] - goal[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def manhattanDistance(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def heuristic_found_by_ga_for_food_problem(start, goal):
    x1, y1 = start
    x2, y2 = goal

    diagonal = diagonal_distance(start, goal)
    max_h = max_heuristic(start, goal)
    min_h = min_heuristic(start, goal)

    return max(diagonal, max_h, min_h)

def aStar(gameState, goal: tuple, heuristic: callable):
    """
    A* algorithm
    """
    start = gameState.getPacmanPosition()
    frontier = util.PriorityQueue()
    frontier.push((start, []), 0)
    explored = set()

    while not frontier.isEmpty():
        current, path = frontier.pop()
        if current == goal:
            return path
        if current not in explored:
            explored.add(current)
            for next in gameState.getLegalActions():
                successor = gameState.generateSuccessor(0, next)
                nextPos = successor.getPacmanPosition()
                if nextPos not in explored:
                    newPath = path + [next]
                    newCost = len(newPath) + heuristic(nextPos, goal)
                    frontier.push((nextPos, newPath), newCost)
    return []

class GeneticAlgorithm:

    def __init__(self, 
                 n_genes,
                 n_iterations,
                 lchrom, 
                 pcross, 
                 pmutation, 
                 selection_type, 
                 popsize, 
                 n_elites,
                 data,
                 MAX_VALUE,
                 MIN_VALUE):
        

        self.n_genes = n_genes
        self.lchrom = lchrom
        self.popsize = popsize
        self.pcross = pcross
        self.pmutation = pmutation
        self.selection_type = selection_type
        self.n_iterations = n_iterations
        self.n_elites = n_elites
        self.data = data
        self.MAX_VALUE = MAX_VALUE
        self.MIN_VALUE = MIN_VALUE
        self.best_fitness_evolution = []
    
        pop = []
        while (len(pop) < self.popsize):
            # generate random chromosome a list of a random number between MIN_VALUE and MAX_VALUE where list length is n_genes
            chromosome = [np.random.randint(self.MIN_VALUE, self.MAX_VALUE) for _ in range(self.n_genes)]
            # if chromosome = set of 0 then generate new one
            if sum(chromosome) == 0:
                continue
            else:
                pop.append(chromosome)

        # Convert pop to list of solutions
        self.population = [tuple(x) for x in pop]
    


    def fitness_func(self, solution):
        # should maximize
        def customEvaluationFunction(currentGameState):
            def DClosestFood(current_pos, foodGrid, ghosts_pos):
                closestFood = foodHeuristic(current_pos, foodGrid)
                if closestFood == 0:
                    closestFood = 1
                # if there is chance (thus manhattanDistance and not exact distance)
                # that there is a ghost nearby dont risk it
                for ghost in ghosts_pos:
                    if manhattanDistance(current_pos, ghost) < 2:
                        closestFood = 99999
                return closestFood
            
            def isNearGhost(current_pos, ghosts_pos):
                # exact distance to ghost
                for ghost in ghosts_pos:
                    estimadedDistance = manhattanDistance(current_pos, ghost)
                    if estimadedDistance < 3:
                        if len(aStar(currentGameState, ghost, manhattanDistance)) < 2:
                            return 99999
                return 0

            current_pos = currentGameState.getPacmanPosition()
            ghosts_pos = currentGameState.getGhostPositions()

            foodGrid = currentGameState.getFood()
            capsuleList = currentGameState.getCapsules()

            features = [1.0/DClosestFood(current_pos, foodGrid, ghosts_pos),
                        currentGameState.getScore(),
                        isNearGhost(current_pos, ghosts_pos),
                        ]

            score = 0
            for i in range(len(features)):
                score += features[i] * solution[i]
            return score

        # All MultiAgentSearchAgent have the same __init__ function 
        # thus we can generlize and pass the agent to the genetic algorithm using command line
        customAgent = self.data.agentToUse(actualEvalFunc=customEvaluationFunction, evalFn="doesntexists", depth=2)
        baseGhosts =  [ghostAgents.DirectionalGhost(i+1) for i in range(self.data.numberOfGhosts)]
        game = self.data.rules.newGame(self.data.layout, customAgent, baseGhosts, self.data.gameDisplay)
        game.run() # simulate the game using a custom evaluation function given by the solution (weights)
        score = game.state.getScore()
        print("Individual score: ", score, " with weights: ", solution, "\n")
        cache = {} # clear cache

        return score

    def get_fitness_scores(self):
        scores = [self.fitness_func(ind) for ind in self.population]
        return np.array(scores)

    def __append_best_score(self, scores):
        best_score = np.max(scores)
        self.best_fitness_evolution.append(best_score)
        return 'Ok'
    
    def __ranking_selection(self, scores):
        ind = np.argsort(scores)
        best_ind = ind[-1]
        return best_ind

    def select(self, scores, selection_type):
        if selection_type not in ['ranking', 'roulette']:
            raise ValueError('Type should be ranking or tournament')

        if selection_type == 'ranking':
            ind = self.__ranking_selection(scores)
        elif selection_type == 'roulette':
            ind = self.__roulette_selection(scores)
        else:
            pass
        return ind

    def flip(self, p):
        return 1 if np.random.rand() < p else 0

    def __crossover(self, 
                    parent1, 
                    parent2, 
                    pcross,
                    lchrom):
        index = np.random.choice(range(1, lchrom)) 
        parent1 = list(parent1)
        parent2 = list(parent2)
        child1 = parent1[:index] + parent2[index:]
        child2 = parent2[:index] + parent1[index:]
        children = [tuple(child1), tuple(child2)]
        return children
    
    def __mutation(self, individual):

        index = np.random.choice(len(individual))
        
        # Convert individual to list so that can be modified
        individual_mod = list(individual)
        individual_mod[index] = np.random.randint(self.MIN_VALUE, self.MAX_VALUE)
        individual = tuple(individual_mod)

        return individual

    def optimize(self):

        for i in tqdm(range(self.n_iterations)):

            # calculate fitness score
            scores = self.get_fitness_scores()

            # choose the elites of the current population
            ind = np.argsort(scores)

            elites = [self.population[i] for i in ind[-self.n_elites:]]

            print("Elites for iteration {} are {}".format(i, elites))
            #append the elites to the population
            new_population = [tuple(elite) for elite in elites]

            # make selection
            j = self.n_elites
            while j <= self.popsize:
                # select parents from population
                mate1 = self.select(scores, self.selection_type)
                mate2 = self.select(scores, self.selection_type)


                mate1 = tuple(self.population[mate1])
                mate2 = tuple(self.population[mate2])

                if self.flip(self.pcross):
                    children = self.__crossover(mate1, mate2, self.pcross, self.lchrom)
                    children = [tuple(child) for child in children]
                else:
                    children = [mate1, mate2]
                
                if self.flip(self.pmutation):
                    children[0] = self.__mutation(children[0])

                if self.flip(self.pmutation):
                    children[1] = self.__mutation(children[1])

                if sum(tuple(children[0])) != 0:
                    new_population.append(tuple(children[0]))
                    j+=1
                
                if sum(tuple(children[1])) != 0:
                    new_population.append(tuple(children[1]))        
                    j+=1

            self.population = new_population

        # when n_iterations are over, fitness scores
        scores = self.get_fitness_scores()

        # append best score
        _ = self.__append_best_score(scores)

        # get the result wher he results is the best
        best_score_ind =np.argpartition(scores, 0)[0]
    
        best_solution = self.population[best_score_ind]

        return (best_solution, self.best_fitness_evolution[-1])

def default(str):
    return str + ' [Default: %default]'

# A wrapper class just to hold data to be sent to the genetic algorithm
class Data:
    def __init__(self, layout, rules, gameDisplay, numberOfGhosts, agentToUse):
        self.layout = layout
        self.rules = rules
        self.gameDisplay = gameDisplay
        self.numberOfGhosts = numberOfGhosts
        self.agentToUse = agentToUse

def main( argv ):
    from optparse import OptionParser
    usageStr = """
    USAGE:      python genetic_algorithm.py <options>
    EXAMPLES:   (1) python genetic_algorithm.py -l smallClassic -k 10
                    - starts genetic algorithm on a small layout with 10 ghosts agents
    """
    parser = OptionParser(usageStr)
    parser.add_option('-k', '--numGhosts', dest='numGhosts', type='int',
                      help=default('the number of ghosts to run'), metavar='GHOSTS', default=10)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='smallClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                        help=default('the agent TYPE in the pacmanAgents module to use'),
                        metavar='TYPE', default='AlphaBetaAgent')
    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    
    used_layout = layout.getLayout(options.layout)
    k = options.numGhosts

    agentToUse = options.pacman

    if (agentToUse not in ["MinimaxAgent", "AlphaBetaAgent", "ExpectimaxAgent"]):
        raise Exception("Invalid agent to use")
    # get refrence to the class based on the string
    agentToUse = getattr(multiAgents, agentToUse)
    rules = pacman.ClassicGameRules()
    gameDisplay = textDisplay.NullGraphics()
    rules.quiet = True
    
    data = Data(used_layout, rules, gameDisplay, k, agentToUse)
    
    ga = GeneticAlgorithm(
        n_genes = 3,
        n_iterations = 10,
        lchrom = 3,
        pcross = 0.8, 
        pmutation = 0.35,
        selection_type = 'ranking', 
        popsize = 20,
        n_elites = 2,
        data = data,
        MAX_VALUE = 1000,
        MIN_VALUE = -1000,
    )

    best_solution, best_fitness = ga.optimize()
    print('\nBest solution:\t', best_solution)
    print('\nBest Fitness:\t', round(best_fitness))
    print("\n\n----------------------------------\n\n")

if __name__ == "__main__":
    main( sys.argv[1:] )
