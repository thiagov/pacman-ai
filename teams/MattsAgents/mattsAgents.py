# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
from game import Actions
from game import Configuration
import keyboardAgents
import game
from util import nearestPoint
import regularMutation


# python capture.py -k 3 -l fastCapture
# python capture.py -r mattsAgents.MattsAgents -b BaselineAgents -k 3

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class MattsAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', third='offense', rest='offense', **args):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second, third]
    self.rest = rest
    self.debug = int(args['debug']) if args.has_key('debug') else 0 
    self.maxPlys = int(args['maxPlys']) if args.has_key('maxPlys') else -1
    self.maxTime = float(args['maxTime']) if args.has_key('maxTime') else 0.9
    self.lastMoveCount = int(args['lastMoveCount']) if args.has_key('lastMoveCount') else 5
    self.nonIterative = args.has_key('nonIterative')
    self.epsilon = args['epsilon'] if args.has_key('epsilon') else 0.0
    self.gamma = args['gamma'] if args.has_key('gamma') else 0.0
    self.alpha = args['alpha'] if args.has_key('alpha') else 0.0

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'offense':
      weightStrategy = OffenseWeightStrategy()
      weightStrategies.append(weightStrategy)
      weightStrategies.sort(key=lambda x: x.getAggressiveness(), reverse=True)
      agent = MattReflexCaptureAgent(index, self.debug, self.maxPlys, self.maxTime, self.lastMoveCount, weightStrategy, self.nonIterative)
      distanceToOpponentFood[agent] = 0.0
      return agent
    elif agentStr == 'defense':
      weightStrategy = DefenseWeightStrategy()
      weightStrategies.append(weightStrategy)
      weightStrategies.sort(key=lambda x: x.getAggressiveness(), reverse=True)
      agent = MattReflexCaptureAgent(index, self.debug, self.maxPlys, self.maxTime, self.lastMoveCount, weightStrategy, self.nonIterative)
      distanceToOpponentFood[agent] = 0.0
      return agent
    else:
      raise Exception("No staff agent identified by " + agentStr)

##########
# Agents #
##########

distancebeliefs = dict()
legalMoves = None
distanceToOpponentFood = {}
weightStrategies = []

class MattBaseAgent(CaptureAgent):
  def __init__(self, index, debug):
    CaptureAgent.__init__(self, index)
    self.debug = debug

  def log(self, level, text):
    if self.debug >= level:
      print text

class MattReflexCaptureAgent(MattBaseAgent):
  def __init__(self, index, debug, maxPlys, maxTime, lastMoveCount, weightStrategy, nonIterative):
    MattBaseAgent.__init__(self, index, debug)
    self.firstTurnComplete = False
    self.startingFood = 0
    self.theirStartingFood = 0
    self.maxPlys = maxPlys
    self.maxTime = maxTime
    self.lastMoveCount = lastMoveCount
    self.weightStrategy = weightStrategy
    self.nonIterative = nonIterative
    self.lastMoves = []
  
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    global distancebeliefs, legalMoves, distanceToOpponentFood

    start = time.time()

    # Update last moves
    self.lastMoves.insert(0, gameState.getAgentPosition(self.index))
    if len(self.lastMoves) > self.lastMoveCount:
      self.lastMoves.pop()

    # Init starting food
    if not self.firstTurnComplete:
      self.firstTurnComplete = True
      self.startingFood = len(self.getFoodYouAreDefending(gameState).asList())
      self.theirStartingFood = len(self.getFood(gameState).asList())

    if legalMoves is None:
      legalMoves = self.getLegalMoves(gameState)

    opponents = self.getOpponents(gameState)

    self.log(1, '%s %d (%s):' % ({True:"Red",False:"Blue"}[self.red], self.index, self.weightStrategy.getName()))

    # update opponent observations for Bayesian inference
    previousAgent = (self.index - 1) % gameState.getNumAgents()
    for opponent in opponents:
      if self.firstTurnComplete and opponent == previousAgent:
        self.elapseTime(gameState, opponent)

      self.observation(gameState, opponent)

    if self.debug >= 1:
      # Display belief distributions
      display = []
      for agentIndex in range(gameState.getNumAgents()):
        if distancebeliefs.has_key(agentIndex):
          display.append(distancebeliefs[agentIndex])
        else:
          display.append(None)
      self.displayDistributionsOverPositions(display)

    # Compute distance to the nearest food
    foodList = self.getFood(gameState).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in foodList])
      distanceToOpponentFood[self] = minDistance

      # Sort agents in ascending order of distance to opponent food,
      # plus a small bias towards the current level of aggressiveness
      agentDist = sorted(distanceToOpponentFood.items(), key=lambda x: (x[0].weightStrategy.getAggressiveness() * -2 + x[1]))
      # Assign weight strategies, with the most aggressive given to the closest to opponent food
      # and the least aggressive given to the furthest away
      for (agent, dist), weightStrategy in zip(agentDist, weightStrategies):
        if agent.weightStrategy.getName() is not weightStrategy.getName():
          self.log(1, 'agent %d at distance %d now has strategy %s' % (agent.index, dist, weightStrategy.getName()))
        agent.weightStrategy = weightStrategy

    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    random.shuffle(actions)

    # Instead of dealing with belief distributions, make a fake game state
    # which just assigns the most likely positions for the opponents
    # so we can do standard minimax
    beliefGameState = gameState.deepCopy()
    for agentIndex in range(beliefGameState.getNumAgents()):
      if distancebeliefs.has_key(agentIndex):
        pos = distancebeliefs[agentIndex].argMax()
        beliefGameState.getAgentState(agentIndex).configuration = Configuration(pos, Directions.STOP)

    if self.debug >= 3:
      # Print the weights and features which are currently affecting the agent
      self.log(3, 'feature, value, weight, total')
      weights = self.getWeights(beliefGameState, Directions.STOP)
      for (feature, value) in self.getFeatures(beliefGameState, Directions.STOP).items():
        weight = 0.0
        if feature in weights:
          weight = weights[feature]
        self.log(3, '%s, %f, %f, %f' % (feature, value, weight, value * weight))

    if self.nonIterative:
      values = [self.evaluateMinimax(beliefGameState, a, self.maxPlys, self.index, "-inf", "+inf", start) for a in actions]
    else:
      """
      Minimax with iterative deepening and alpha-beta pruning.
      """
      depth = 0
      while True:
        try:
          # Start minimax for the given ply depth, with alpha=-inf and beta=+inf,
          # including the action start time so we can quit before the timeout
          values = [self.evaluateMinimax(beliefGameState, a, depth, self.index, float('-inf'), float('+inf'), start) for a in actions]
          if self.debug >= 3:
            # Log the action we would take if we stopped at this level
            maxValue = reduce(max, values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]

            action = random.choice(bestActions)

            self.log(3, 'eval time for agent %d at depth %d: %.4f (action %s)' % (self.index, depth, time.time() - start, action))
          else:
            self.log(2, 'eval time for agent %d at depth %d: %.4f' % (self.index, depth, time.time() - start))
        except Exception:
          # We timed out; use the previous level's values rather than incomplete data
          self.log(2, 'throttled agent %d at depth %d: %.4f' % (self.index, depth, time.time() - start))
          break
        # We must go deeper
        depth = depth + 2
        if (self.maxPlys > -1) and depth > self.maxPlys:
          # We must not go deeper
          self.log(2, 'max plys reached for agent %d at depth %d: %.4f' % (self.index, depth, time.time() - start))
          break

    # Determine the best possible Q() and all actions which achieve it,
    # and pick one of those at random
    maxValue = reduce(max, values)
    actionsValues = zip(actions, values)
    bestActions = [a for a, v in actionsValues if v == maxValue]

    action = random.choice(bestActions)
    self.log(1, actionsValues)
    self.log(1, 'agent %d selected action %s with value %s (time %.4f)' % (self.index, action, maxValue, time.time() - start))
    return action

  def getLegalMoves(self, gameState):
    """
    Returns a list of every space that is not a wall.
    """
    global legalMoves
    if legalMoves is not None:
      return legalMoves

    legalMoves = []
    for x in range(gameState.data.layout.width):
      for y in range(gameState.data.layout.height):
        if gameState.hasWall(x, y): continue
        legalMoves.append((x, y))
    return legalMoves

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluateMinimax(self, gameState, action, plysLeft, agentIndex, alpha, beta, startTime):
    """
    Runs the minimax algorithm with alpha-beta pruning.
    Will throw an exception if it has been too long since the start time.
    """
    self.log(4, 'agent %d action %s plysLeft %d alpha %s beta %s' % (agentIndex, action, plysLeft, alpha, beta))
    # Various terminal states
    if time.time() - startTime > self.maxTime:
      raise Exception("Too long")
    if gameState.isOver():
      if gameState.getScore() == 0:
        return 0
      elif (self.red and gameState.getScore() > 0) or (not self.red and gameState.getScore() < 0):
        return float('+inf')
      else:
        return float('-inf')

    actedGameState = gameState.generateSuccessor(agentIndex, action)
    """ Check if we died by someone else stepping on us. """
    if agentIndex != self.index and self.getPosition(gameState) != self.getPosition(actedGameState):
      return float('-inf')

    if plysLeft == 0:
      value = self.evaluateTerminal(gameState, action)
      self.log(4, 'terminal value %f' % (value))
      return value

    # Perform the given action
    actedGameState = gameState.generateSuccessor(agentIndex, action)
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
    legalActions = actedGameState.getLegalActions(nextAgentIndex)

    # If the next agent is very far away, it's not worth the extra time to consider his actions; just assume he won't move
    if self.distancer.getDistance(gameState.getAgentPosition(agentIndex), gameState.getAgentPosition(nextAgentIndex)) > (plysLeft * 3):
      self.log(4, 'agent %d is too far away' % (nextAgentIndex))
      legalActions = [Directions.STOP]

    if gameState.isOnRedTeam(agentIndex) == gameState.isOnRedTeam(self.index):
      # Max node
      maxValue = alpha
      for act in legalActions:
        stateValue = self.evaluateMinimax(actedGameState, act, plysLeft - 1, nextAgentIndex, maxValue, beta, startTime)
        maxValue = max(maxValue, stateValue)
        if maxValue >= beta:
          self.log(4, 'truncate at plysLeft %d due to beta %s' % (plysLeft, beta))
          break
      return maxValue
    else:
      # Min node
      minValue = beta
      for act in legalActions:
        stateValue = self.evaluateMinimax(actedGameState, act, plysLeft - 1, nextAgentIndex, alpha, minValue, startTime)
        minValue = min(minValue, stateValue)
        if minValue <= alpha:
          self.log(4, 'truncate at plysLeft %d due to alpha %s' % (plysLeft, alpha))
          break
      return minValue

  def evaluateTerminal(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return self.weightStrategy.getWeights(gameState, action)
  
  """
  Features (not the best features) which have learned weight values stored.
  """
  def getMutationFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    position = self.getPosition(gameState)

    distances = 0.0
    for tpos in self.getTeamPositions(successor):
      distances = distances + abs(tpos[0] - position[0])
    features['xRelativeToFriends'] = distances
    
    enemyX = 0.0
    for epos in self.getOpponentPositions(successor):
      if epos is not None:
        enemyX = enemyX + epos[0]
    features['avgEnemyX'] = distances
    
    foodLeft = len(self.getFoodYouAreDefending(successor).asList())
    features['percentOurFoodLeft'] = foodLeft / self.startingFood
    
    foodLeft = len(self.getFood(successor).asList())
    features['percentTheirFoodLeft'] = foodLeft / self.theirStartingFood
    
    features['IAmAScaredGhost'] = 1.0 if self.isPacman(successor) and self.getScaredTimer(successor) > 0 else 0.0
    
    features['enemyPacmanNearMe'] = 0.0
    minOppDist = 10000
    minOppPos = (0, 0)
    for ep in self.getOpponentPositions(successor):
      # For a feature later on
      if ep is None:
        continue
      enemyDistance = self.getMazeDistance(ep, position)
      if enemyDistance < minOppDist:
        minOppDist = self.getMazeDistance(ep, position)
        minOppPos = ep
      if enemyDistance <= 1:
        if self.isPositionInEnemyTerritory(successor, ep) or self.getScaredTimer(successor) > 0:
          features['attackerNearMe'] = 1.0
        else:
          features['enemyPacmanNearMe'] = 1.0

    myPos = successor.getAgentState(self.index).getPosition()

    features['numSameFriends'] = 0
    closestSameFriend = None
    distanceY = 0.0
    for friend in self.getTeam(successor):
      # Don't count ourselves (so lonely)
      if friend == self.index:
        continue
      if successor.getAgentPosition(friend) == myPos:
        features['onTopOfTeammate'] = 1
      if successor.getAgentState(friend).isPacman is self.isPacman(successor):
        friendPosition = successor.getAgentPosition(friend)
        friendDistance = self.getMazeDistance(position, friendPosition)
        distanceY += abs(myPos[1] - friendPosition[1])
        if closestSameFriend is None:
          closestSameFriend = friendDistance
        else:
          closestSameFriend = min(closestSameFriend, friendDistance)

        features['numSameFriends'] = features['numSameFriends'] + 1

    if closestSameFriend is not None:
      features['yRelativeToFriends'] = distanceY
      features['closestSameFriend'] = closestSameFriend

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDiffDistance = min([1000] + [self.getMazeDistance(position, food) - self.getMazeDistance(minOppPos, food) for food in foodList if minOppDist < 1000])
      features['blockableFood'] = 1.0 if minDiffDistance < 1.0 else 0.0

    # Added by Kenny
    distributions = [beliefs for i,beliefs
                     in distancebeliefs.items()
                     if gameState.isOnRedTeam(i) is not self.red]
    if len(distributions) > 0:
      onOurSide = self.isPositionInTeamTerritory(gameState, myPos)
      ourSideDistance, theirSideDistance = None, None
      for belief in distributions:
        mostLikelyPos = belief.argMax()
        distance = self.distancer.getDistance(myPos, mostLikelyPos)
        if self.isPositionInEnemyTerritory(gameState, mostLikelyPos):
          if onOurSide:
            continue
          if theirSideDistance is None:
            theirSideDistance = distance
          else:
            theirSideDistance = min(theirSideDistance, distance)
        else:
          if not onOurSide:
            continue
          if ourSideDistance is None:
            ourSideDistance = distance
          else:
            ourSideDistance = min(ourSideDistance, distance)

      if ourSideDistance is not None or theirSideDistance is not None:
        if ourSideDistance is None:
          closestProbableEnemy = theirSideDistance
        elif theirSideDistance is None:
          closestProbableEnemy = ourSideDistance
        else:
          closestProbableEnemy = min(ourSideDistance, theirSideDistance)

        features['closestProbableEnemy'] = closestProbableEnemy

        if ourSideDistance is not None:
          features['closestProbableEnemyOurSide'] = ourSideDistance
          features['closestProbableEnemyOurSideRecip'] = 1 / (ourSideDistance + 1)

        if theirSideDistance is not None:
          features['closestProbableEnemyTheirSide'] = theirSideDistance
          features['closestProbableEnemyTheirSideRecip'] = 1 / (theirSideDistance + 1)

    if action == Directions.STOP: features['stop'] = 1

    return features

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = self.getMutationFeatures(gameState, action)
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    if (myPos in self.lastMoves):
      features['oldPosition'] = self.lastMoveCount - self.lastMoves.index(myPos)

    features['successorScore'] = self.getScore(successor)
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    features['numFood'] = len(foodList)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['minDistanceToOppFood'] = minDistance

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    foodList = self.getFoodYouAreDefending(successor).asList()
    distance = 0
    for food in foodList:
      distance = distance + self.getMazeDistance(myPos, food)
    features['totalDistanceToOwnFood'] = distance

    return features


  def observation(self, gameState, agentIndex):
    agentPosition = gameState.getAgentPosition(agentIndex)

    # If we know where they are, we don't have to guess.
    if agentPosition is not None:
      beliefs = util.Counter()
      beliefs[agentPosition] = 1.0
      self.setBeliefsForAgent(agentIndex, beliefs)
      self.log(1, '  agent %d: position=%s' % (agentIndex, agentPosition))
      return

    oldBeliefs = self.getBeliefsForAgent(gameState, agentIndex)

    #currentAgentPosition = self.getPositionAsInt(gameState)
    teste = gameState.getAgentState(self.index).getPosition()
    currentAgentPosition = (int(teste[0]), int(teste[1]))
    # Make the noisy distance observation
    noisyDistance = gameState.getAgentDistances()[agentIndex]

    # Update belief distribution with Bayesian inference
    beliefs = util.Counter()
    for p in self.getLegalMoves(gameState):
      trueDistance = util.manhattanDistance(p, currentAgentPosition)
      """
      Compute:
      Pr(agent at p | previous game state) * Pr(observation | (agent at p | previous game state)) =
      Pr(observation and (agent at p | previous game state)) =
      Pr(observation) * Pr((agent at p | previous game state) | observation) =
      Pr(observation) * Pr(agent at p | (previous game state and observation)) =
      Pr(observation) * Pr(agent at p | current game state)
      """
      # Pr(agent at p | previous game state) * Pr(observation | (agent at p | previous game state))
      beliefs[p] = gameState.getDistanceProb(trueDistance, noisyDistance) * oldBeliefs[p]
    """
    beliefs now contains Pr(observation) * Pr(agent at p | current game state) for all p.

    Pr(agent at p | current game state) for all p is the distribution we actually want,
    so factor out Pr(observation) by normalizing the distribution
    (alternatively, set it equal to 1 since it actually happened).
    """
    beliefs.normalize()

    self.log(1, '  agent %d: noisy distance=%d' % (agentIndex, noisyDistance))
    self.setBeliefsForAgent(agentIndex, beliefs)

  def getPositionDistribution(self, gameState, oldPos):
    # Agents can stay still, or move to an adjacent non-wall space
    dist = util.Counter()
    dist[oldPos] = 1.0
    for a in Actions.getLegalNeighbors(oldPos, gameState.getWalls()): dist[a] = 1.0
    dist.normalize()
    return dist

  def elapseTime(self, gameState, agentIndex):
    oldBeliefs = self.getBeliefsForAgent(gameState, agentIndex)

    allPossible = util.Counter()
    for p in self.getLegalMoves(gameState):
      # Push the old probability beliefs out by one move in all directions (including stop)
      newPosDist = self.getPositionDistribution(gameState, p)
      for newPos, prob in newPosDist.items():
        allPossible[newPos] += prob * oldBeliefs[p]
    allPossible.normalize()

    self.setBeliefsForAgent(agentIndex, allPossible)

  def getBeliefsForAgent(self, gameState, agentIndex):
    global distancebeliefs

    if distancebeliefs.has_key(agentIndex):
      beliefs = distancebeliefs[agentIndex]

      # If our beliefs have reduced to zero, reinitialize.
      if beliefs.totalCount() != 0:
        return beliefs

    # Initialize to uniform belief across all legal spaces
    beliefs = util.Counter()
    for p in self.getLegalMoves(gameState):
      beliefs[p] = 1.0
    beliefs.normalize()
    self.log(1, "--inited--")

    return beliefs

  def setBeliefsForAgent(self, agentIndex, beliefs):
    global distancebeliefs
    distancebeliefs[agentIndex] = beliefs

class OffenseWeightStrategy:
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getWeights(self, gameState, action):
    weights = regularMutation.aggressiveDWeightsDict
    weights['onDefense'] = -20
    weights['successorScore'] = 1.5
    # Always eat nearby food
    weights['numFood'] = -1000
    # Favor reaching new food the most
    weights['minDistanceToOppFood'] = -10
    # Avoid real enemies
    weights['invaderDistance'] = -0.5
    # Avoid probable enemies on their side
    weights['closestProbableEnemyTheirSideRecip'] = 25
    # Try to cover more board with multiple teammates
    weights['closestSameFriend'] = 5
    weights['stop'] = -1
    weights['onTopOfTeammate'] = 0.2
    weights['yRelativeToFriends'] = 0.2
    weights['attackerNearMe'] = 1
    weights['oldPosition'] = -2

    return weights

  def getAggressiveness(self):
    return 1

  def getName(self):
    return "offense"

class DefenseWeightStrategy:
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def getWeights(self, gameState, action):
    weights = regularMutation.goalieDWeightsDict
    weights['sucessorScore'] = 100
    weights['numInvaders'] = -100
    weights['onDefense'] = 100
    weights['invaderDistance'] = -1.5
    weights['totalDistanceToOwnFood'] = -0.4
    weights['stop'] = -1
    weights['reverse'] = -1
    weights['closestProbableEnemyOurSide'] = -1.5
    weights['closestProbableEnemy'] = -1.5
    weights['onTopOfTeammate'] = 0.1
    weights['closestProbableEnemyOurSideRecip'] = 50
    weights['oldPosition'] = -2
    return weights

  def getAggressiveness(self):
    return 0

  def getName(self):
    return "defense"
