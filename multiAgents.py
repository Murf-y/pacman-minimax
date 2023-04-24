# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from pacman import GameState
import math

# python pacman.py -l smallClassic -k 7 -p ReflexAgent
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, index = 0):
        self.index = index
        self.cache = {}

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        currentFood = currentGameState.getFood()
        # Initialize the heuristic value to zero
        heuristicValue = 0
        # Check if the game is over
        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")
        # Compute the number of food pellets in the current and successor states
        numCurrentFood = currentFood.count()
        numSuccessorFood = newFood.count()
        min_food_distance = float("inf")
        sum_ghost_distances = 0
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentGhostPositions = currentGameState.getGhostPositions()
        # Compute the distance to the closest food pellet
        for food in newFood.asList():
            min_food_distance = min(min_food_distance, manhattanDistance(newPos, food))
        
        # Compute the distance to the closest ghost
        for ghost in newGhostStates:
            sum_ghost_distances += manhattanDistance(newPos, ghost.getPosition())

        # Compute the heuristic value
        score = 88 * (currentGameState.getNumFood() - successorGameState.getNumFood()) + 89 * (1 / min_food_distance) + -96 * (1 / sum_ghost_distances) + 70 * sum(newScaredTimes) + -60 * (1 if newPos in currentGhostPositions else 0) + (-1000 if action == Directions.STOP else 0)
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, actualEvalFunc=lambda x: 0, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        try:
            self.evaluationFunction = util.lookup(evalFn, globals())
        except:
            self.evaluationFunction = actualEvalFunc
            
        self.depth = int(depth)

# python pacman.py -l smallClassic -k 7 -p MinimaxAgent 
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.minValue(gameState, depth, agentIndex)
        
    def maxValue(self, state, currentDepth, agentIndex):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action
        return v, bestAction
    
    def minValue(self, state, currentDepth, agentIndex):
        v = float("inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.minimax(successor, successorIndex, successorDepth)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action
        return v, bestAction
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        bestScore, bestAction = self.minimax(gameState, 0, 0)

        return bestAction

# python pacman.py -p AlphaBetaAgent -l smallClassic -k 10 -a depth=2,evalFn=better --frameTime 0 -q -n 5
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentIndex, alpha, beta)
    
    def maxValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)
        return v, bestAction
    
    def minValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue < v:
                v = successorValue
                bestAction = action

            if v < alpha:
                return v, bestAction

            beta = min(beta, v)
        return v, bestAction
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))

        return bestAction

# python pacman.py -p ExpectimaxAgent -l smallClassic -k 10 -a depth=2,evalFn=better --frameTime 0 -q -n 5
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.expValue(gameState, depth, agentIndex)
        
    def maxValue(self, state, currentDepth, agentIndex):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.expectimax(successor, successorIndex, successorDepth)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action
        return v, bestAction
    
    def expValue(self, state, currentDepth, agentIndex):
        v = 0
        bestAction = None
        allActions = state.getLegalActions(agentIndex)

        if len(allActions) == 0:
            return self.evaluationFunction(state), None
        successorProb = 1 / len(allActions)

        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.expectimax(successor, successorIndex, successorDepth)[0]

            v += successorValue
        v /= len(allActions)
        return v, bestAction
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        bestScore, bestAction = self.expectimax(gameState, 0, 0)

        return bestAction

# python pacman.py -p ExpectimaxAlphaBetaPruningAgent -l smallClassic -k 10 -a depth=2,evalFn=better --frameTime 0 -q -n 5
class ExpectimaxAlphaBetaPruningAgent(MultiAgentSearchAgent):
    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.expValue(gameState, depth, agentIndex, alpha, beta)
    
    def maxValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = float("-inf")
        bestAction = None
        allActions = state.getLegalActions(agentIndex)
        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                v = successorValue
                bestAction = action

            if v > beta:
                return v, bestAction

            alpha = max(alpha, v)
        return v, bestAction
    
    def expValue(self, state, currentDepth, agentIndex, alpha, beta):
        v = 0
        bestAction = None
        allActions = state.getLegalActions(agentIndex)

        if len(allActions) == 0:
            return self.evaluationFunction(state), None
        successorProb = 1 / len(allActions)

        for action in allActions:
            successor = state.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = currentDepth

            if successorIndex == state.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            v += successorValue
        v /= len(allActions)
        return v, bestAction
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        bestScore, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))

        return bestAction

def aStar(gameState: GameState, goal: tuple, heuristic: callable):
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
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    solution = [
        124,
        319,
        -500,
    ]

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

# Abbreviation
better = betterEvaluationFunction

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

def heuristic_found_by_ga_for_food_problem(start, goal):
    x1, y1 = start
    x2, y2 = goal

    diagonal = diagonal_distance(start, goal)
    max_h = max_heuristic(start, goal)
    min_h = min_heuristic(start, goal)

    return max(diagonal, max_h, min_h)

