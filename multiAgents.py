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
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

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
        200,
        162,
        -162,
        453
    ]

    def DClosestFood(current_pos, foodGrid, ghosts_pos):
        closestFood = 1
        food_distances = [manhattanDistance(current_pos, food) for food in foodGrid.asList()]
        
        if len(food_distances) > 0:
            closestFood = min(food_distances)
        # if food was near a ghost, then better not go there
        for ghost in ghosts_pos:
            if manhattanDistance(current_pos, ghost) < 2:
                closestFood = 99999
        return closestFood
    
    def isNearGhost(current_pos, ghosts_pos):
        for ghost in ghosts_pos:
            if manhattanDistance(current_pos, ghost) < 2:
                return -99999
        return 0
    
    def numberOfNonScaredGhosts(currentGameState, ghostStates):
        numberOfNonScaredGhosts = 0
        for ghost in ghostStates:
            if ghost.scaredTimer == 0:
                numberOfNonScaredGhosts += 1
        return numberOfNonScaredGhosts

    current_pos = currentGameState.getPacmanPosition()
    ghosts_pos = currentGameState.getGhostPositions()

    foodGrid = currentGameState.getFood()
    capsuleList = currentGameState.getCapsules()

    numberOfFood = foodGrid.count()
    if numberOfFood == 0:
        numberOfFood = 1
    numberOfCapsules = len(capsuleList)
    if numberOfCapsules == 0:
        numberOfCapsules = 1

    features = [1.0 / DClosestFood(current_pos, foodGrid, ghosts_pos),
                currentGameState.getScore(),
                isNearGhost(current_pos, ghosts_pos),
                numberOfNonScaredGhosts(currentGameState, currentGameState.getGhostStates()),
                1/numberOfFood,
                1/numberOfCapsules]

    score = 0
    for i in range(len(features)):
        score += features[i] * solution[i]
    return score

# Abbreviation
better = betterEvaluationFunction

            # solution = [float(x) / sum(solution) for x in solution]

            # a = solution[0]
            # b = solution[1]
            # c = solution[2]
            # d = solution[3]
            # e = solution[4]
            # f = solution[5]
            # g = solution[6]
            # h = solution[7]
            # i = solution[8]

            # def DClosestFood(current_pos, foodGrid):
            #     closestFood = 1
            #     for food in foodGrid.asList():
            #         closestFood = min(closestFood, manhattanDistance(current_pos, food))
            #     return closestFood
            
            # def DClosestGhost(current_pos, ghostStates):
            #     # distance to the closest non scared ghost
            #     closestGhost = 1
            #     for ghost in ghostStates:
            #         if ghost.scaredTimer == 0:
            #             closestGhost = min(closestGhost, manhattanDistance(current_pos, ghost.getPosition()))
            #     return closestGhost
            
            # def DClosestCapsule(current_pos, capsuleList):
            #     closestCapsule = 1
            #     for capsule in capsuleList:
            #         closestCapsule = min(closestCapsule, manhattanDistance(current_pos, capsule))
            #     return closestCapsule
            
            # def numScaredGhosts(current_pos, ghostStates):
            #     numGhosts = 1
            #     for ghost in ghostStates:
            #         if ghost.scaredTimer > 0 and manhattanDistance(current_pos, ghost.getPosition()) < 2:
            #             numGhosts += 1
            #     return numGhosts
            
            # def numNonScaredGhosts(current_pos, ghostStates):
            #     numGhosts = 0
            #     for ghost in ghostStates:
            #         if ghost.scaredTimer == 0 and manhattanDistance(current_pos, ghost.getPosition()) < 2:
            #             numGhosts += 1
            #     return numGhosts
            
            # def isItWorthToEatTheClosestFood(current_pos, foodGrid, ghostStates):
            #     closestFood = DClosestFood(current_pos, foodGrid)
            #     closestGhost = DClosestGhost(current_pos, ghostStates)
            #     if closestFood < closestGhost:
            #         return 1
            #     return 0
            
            # def isItWorthToEatTheClosestCapsule(current_pos, capsuleList, ghostStates):
            #     closestCapsule = DClosestCapsule(current_pos, capsuleList)
            #     closestGhost = DClosestGhost(current_pos, ghostStates)
            #     if closestCapsule < closestGhost:
            #         return 1
            #     return 0            

            # def isThereGhostNearby(current_pos, ghostStates):
            #     for ghost in ghostStates:
            #         if manhattanDistance(current_pos, ghost.getPosition()) < 2:
            #             return 1
            #     return 0
            
            # current_pos = currentGameState.getPacmanPosition()
            # foodGrid = currentGameState.getFood()
            # ghostStates = currentGameState.getGhostStates()
            # capsuleList = currentGameState.getCapsules()
            # score = 0

            # # add the features to the score with their weights
            # score += (DClosestFood(current_pos, foodGrid)) * a
            # score += (DClosestCapsule(current_pos, capsuleList)) * c
            # score += (DClosestGhost(current_pos, ghostStates)) * b
            # score += (isItWorthToEatTheClosestFood(current_pos, foodGrid, ghostStates)) * f
            # score += (isItWorthToEatTheClosestCapsule(current_pos, capsuleList, ghostStates)) * g
            # score += (isThereGhostNearby(current_pos, ghostStates)) * h
            # score += (currentGameState.getScore()) * i

            # return score