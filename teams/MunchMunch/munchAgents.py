from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint

from capture import SONAR_NOISE_RANGE
from capture import SONAR_NOISE_VALUES

SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2
SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in SONAR_NOISE_VALUES]

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class MunchAgentFactory(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

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
      return OffensiveReflexAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveReflexAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveReflexAgent(index)
    else:
      return DefensiveReflexAgent(index)

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    
    A distanceCalculator instance caches the maze distances 
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    """
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.enemyIndices = self.getOpponents(gameState)
    
    # comment this out to forgo maze distance computation and use manhattan distances
    self.distancer.getMazeDistances()
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display
    jointInference.initialize(gameState, self.legalPositions, self.enemyIndices, 300)
    self.beliefs = util.Counter()  
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    print gameState.getAgentPosition(1)
    
    # Update our beliefs
    observation = gameState.getAgentDistances()
    jointInference.elapseTime(gameState)
    jointInference.observeState(gameState, self.index)
    agentBeliefs = []
    for i in range(gameState.getNumAgents()):
      if i in self.enemyIndices:
        partial = util.Counter()
        index = self.enemyIndices.index(i)
        for pos_tuple in jointInference.getBeliefDistribution().keys():
          partial[pos_tuple[index]] = jointInference.getBeliefDistribution()[pos_tuple]
        agentBeliefs.insert(i, partial)
      else:
        agentBeliefs.insert(i, None)
    self.displayDistributionsOverPositions(agentBeliefs)
    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  #def registerInitialState(self, gameState):
  #  print "Registering"
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

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

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


### Enemy Tracking ###

# Use Joint Particle Filtering to track each enemy.

class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."
  
  def initialize(self, gameState, legalPositions, enemyTeamIndices, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numEnemy = len(enemyTeamIndices)
    self.numParticles = numParticles
    self.enemyIndices = enemyTeamIndices
    self.friendlyIndices = []
    for i in range(gameState.getNumAgents()):
      if i not in self.enemyIndices:
        self.friendlyIndices.append(i)
    self.legalPositions = legalPositions
    self.initializeParticles()

  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions."
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numEnemy)]))

  def addEnemyAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.enemyIndices.append(agent)
    
  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.
    
    You will need to use two helper methods provided below:
      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.
      
      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions 
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent) 
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          Remember: ghosts start at index 1 (Pacman is agent 0).  
          
          The ghost agent you are meant to supply is self.enemyAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      for enemyIndex in range(len(self.enemyIndices)):
        tmpState = setEnemyPositions(gameState, newParticle, self.enemyIndices)    
        updatedParticle = util.sample(getPositionDistributionForEnemy(tmpState, self.enemyIndices[enemyIndex], self.enemyIndices[enemyIndex]))
        newParticle[enemyIndex] = updatedParticle
      newParticles.append(tuple(newParticle))
    self.particles = newParticles
  
  def observeState(self, gameState, friendlyIndex):
    """
    Resamples the set of particles using the likelihood of the noisy observations.
    
    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
          that the ghost appears in its cell, position (2 * ghostIndex - 1, 1).
          Captured ghosts always have a noisyDistance of 999.
         
      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles.  
    """ 
    friendlyPosition = gameState.getAgentPosition(friendlyIndex)   # Should NEVER be 'None'
    noisyDistances = list()
    #for i in range(gameState.getNumAgents()):
    #  print gameState.getAgentPosition(i)
    for i in range(len(gameState.getAgentDistances())):
      if i in self.enemyIndices:
        noisyDistances.append(gameState.getAgentDistances()[i])
    emissionModels = [getObservationDistribution(dist) for dist in noisyDistances]
    particleWeightsCounter = util.Counter()
    if (len(emissionModels) != 0):
      for particleArray in self.particles:
          particleArrayList = list(particleArray)
          particle_weights = util.Counter()
          for enemyIndex in range(self.numEnemy):
            trueDist = util.manhattanDistance(friendlyPosition, particleArrayList[enemyIndex])
            weight = (float) (emissionModels[enemyIndex][trueDist])
            particle_weights[enemyIndex] += weight
          weightProduct = 1
          for particle in particle_weights:
            weightProduct *= particle_weights[particle]
          particleArrayList = tuple(particleArrayList)
          particleWeightsCounter[particleArrayList] = weightProduct + particleWeightsCounter[particleArrayList]
      if (sum(particleWeightsCounter.values()) == 0):
        self.initializeParticles()
        return
      newParticles = []
      for n in range(len(self.particles)):
        resample = util.sampleFromCounter(particleWeightsCounter)
        newParticles.append(resample)
      self.particles = newParticles;
    
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference 
jointInference = JointParticleFilter()

def getPositionDistributionForEnemy(gameState, enemyIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """
  enemyPosition = gameState.getAgentPosition(enemyIndex)
  actionDist = getEnemyDistribution(gameState, enemyIndex)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(enemyPosition, action)
    dist[successorPosition] = prob
  return dist

def getEnemyDistribution(gameState, enemyIndex):
  result = util.Counter()
  actions = gameState.getLegalActions(enemyIndex)
  prob = 1.0 / len( actions )
  for action in actions:
    result[action] = prob
  return result

def setEnemyPositions(gameState, ghostPositions, enemyIndices):
  "Sets the position of all ghosts to the values in ghostPositionTuple."
  for i in range(len(ghostPositions)):
    conf = game.Configuration(ghostPositions[i], game.Directions.STOP)
    gameState.data.agentStates[enemyIndices[i]] = game.AgentState(conf, False)
  return gameState  

observationDistributions = {}
def getObservationDistribution(noisyDistance):
  """
  Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
  conditioned upon all the possible true distances that could have generated it.
  """
  global observationDistributions
  if noisyDistance not in observationDistributions:
    distribution = util.Counter()
    for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
      distribution[max(1, noisyDistance - error)] += prob
    observationDistributions[noisyDistance] = distribution
  return observationDistributions[noisyDistance]

