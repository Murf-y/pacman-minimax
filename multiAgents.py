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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
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

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    def closest_dot(cur_pos, food_pos):
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def closest_ghost(cur_pos, ghosts):
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1


    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        num_ghosts = 0
        for ghost in ghost_states:
            if util.manhattanDistance(ghost.getPosition(), cur_pos) <= radius:
                scores -= 30
                num_ghosts += 1
        return scores

    def food_stuff(cur_pos, food_positions):
        food_distances = []
        for food in food_positions:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return sum(food_distances)

    def num_food(cur_pos, food):
        return len(food)

    def closest_capsule(cur_pos, caps_pos):
        capsule_distances = []
        for caps in caps_pos:
            capsule_distances.append(util.manhattanDistance(caps, cur_pos))
        return min(capsule_distances) if len(capsule_distances) > 0 else 9999999

    def scaredghosts(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 4:
                scoreslist.append(scores + 50)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 3:
                scoreslist.append(scores + 60)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 2:
                scoreslist.append(scores + 70)
            if ghost.scaredTimer > 8 and util.manhattanDistance(ghost.getPosition(), cur_pos) <= 1:
                scoreslist.append(scores + 90)
            #if ghost.scaredTimer > 0 and util.manhattanDistance(ghost.getPosition(), cur_pos) < 1:
 #              scoreslist.append(scores + 100)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def ghostattack(ghost_states, cur_pos, scores):
        scoreslist = []
        for ghost in ghost_states:
            if ghost.scaredTimer == 0:
                scoreslist.append(scores - util.manhattanDistance(ghost.getPosition(), cur_pos) - 10)
        return max(scoreslist) if len(scoreslist) > 0 else scores

    def scoreagent(cur_pos, food_pos, ghost_states, caps_pos, score):
        if closest_capsule(cur_pos, caps_pos) < closest_ghost(cur_pos, ghost_states):
            return score + 40
        if closest_dot(cur_pos, food_pos) < closest_ghost(cur_pos, ghost_states) + 3:
            return score + 20
        if closest_capsule(cur_pos, caps_pos) < closest_dot(cur_pos, food_pos) + 3:
            return score + 30
        else:
            return score


    capsule_pos = currentGameState.getCapsules()
    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    #score = score * 2 if closest_dot(pacman_pos, food) < closest_ghost(pacman_pos, ghosts) + 3 else score
    #score = score * 1.5 if closest_capsule(pacman_pos, capsule_pos) < closest_dot(pacman_pos, food) + 4 else score
    score = scoreagent(pacman_pos, food, ghosts, capsule_pos, score)
    score = scaredghosts(ghosts, pacman_pos, score)
    score = ghostattack(ghosts, pacman_pos, score)
    score -= .35 * food_stuff(pacman_pos, food)
    return score

# Abbreviation
better = betterEvaluationFunction
