from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
import random, time, util
from util import nearestPoint

#############
# FACTORIES #
###############################################
# Instanciam os agentes no inicio da partida. #
# Devem estender a classe base AgentFactory.  #
###############################################

class ThiagovTeamFactory(AgentFactory):
  "Gera um time ThiagovTeam"

  def __init__(self, isRed):
    AgentFactory.__init__(self, isRed)        
    self.agentList = ['attacker', 'defender']

  def getAgent(self, index):                 
    if len(self.agentList) > 0:              
      agent = self.agentList.pop(0)           
      if agent == 'attacker':                 
        return Attacker(index)                
    return Defender(index)                   

#############
#  AGENTS   #
###############################################
# Implementacoes dos agentes.                 #
# Devem estender a classe base CaptureAgent.  #
###############################################

class Attacker(CaptureAgent):
  "Agente ofensivo simples."

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

    # Compute distance to the nearest capsule
    oldCapsuleList = self.getCapsules(gameState)
    if len(oldCapsuleList) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      oldMinPosition = min(oldCapsuleList, key = lambda x: self.getMazeDistance(myPos, x))
      if oldMinPosition == myPos:
        features['distanceToCapsule'] = 0
      else:
        capsuleList = self.getCapsules(successor)
        if len(capsuleList) > 0:
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
          features['distanceToCapsule'] = minDistance

    # Compute distance to closest ghost
    myPos = successor.getAgentState(self.index).getPosition()
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    in_range = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(in_range) > 0:
      positions = [agent.getPosition() for agent in in_range]
      closest = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      closestDist = self.getMazeDistance(myPos, closest)
      if closestDist <= 5:
        features['distanceToGhost'] = closestDist

    # Verify if actions is to stay put
    features['isStop'] = 1 if action == Directions.STOP else 0

    # Compute distance to ally
    team = self.getTeam(successor)
    team.remove(self.index)
    ally = successor.getAgentState(team[0])
    ally_position = ally.getPosition()
    ally_dist = self.getMazeDistance(myPos, ally_position)
    features['distanceToAlly'] = ally_dist

    return features

  def getWeights(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    in_range = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(in_range) > 0:
      positions = [agent.getPosition() for agent in in_range]
      closestPos = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      closestDist = self.getMazeDistance(myPos, closestPos)
      closest_enemies = filter(lambda x: x[0] == closestPos, zip(positions, in_range))
      for agent in closest_enemies:
        if agent[1].scaredTimer > 0:
          # If opponent is scared, the agent should not care about distanceToGhost.
          return {'successorScore': 100, 'distanceToFood': -3, 'distanceToCapsule': 0, 'distanceToGhost': 0, 'distanceToAlly': 2, 'isStop': 0}
      if closestDist <= 5 and successor.getAgentState(self.index).isPacman:
        # If agent is being persued, the agent should focus on surviving: it should
        # give priority to distanceToCapsule and distanceToGhost, and it should not stay put
        return {'successorScore': 5, 'distanceToFood': -1, 'distanceToCapsule': -50, 'distanceToGhost': 100, 'distanceToAlly': 0, 'isStop': -200}
    # Normal weights
    return {'successorScore': 100, 'distanceToFood': -3, 'distanceToCapsule': 0, 'distanceToGhost': 1, 'distanceToAlly': 2, 'isStop': 0}

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

  def cutoffTest(self, gameState, depth):
    if depth == 0 or gameState.isOver():
      return True
    return False

  def minimax(self, gameState, action, depth, agentIndex, alpha, beta):
    if self.cutoffTest(gameState, depth):
      return self.evaluate(gameState, action)

    try:
      # Perform the given action
      actedGameState = gameState.generateSuccessor(agentIndex, action)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      legalActions = actedGameState.getLegalActions(nextAgentIndex)
      # If the next agent is very far away, it's not worth the extra time to consider his actions; just assume he won't move
      if self.distancer.getDistance(gameState.getAgentPosition(agentIndex), gameState.getAgentPosition(nextAgentIndex)) > (depth * 3):
        legalActions = [Directions.STOP]
    except:
      actedGameState = gameState
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      legalActions = [Directions.STOP]

    if gameState.isOnRedTeam(agentIndex) == gameState.isOnRedTeam(self.index):
      # Max node
      maxValue = alpha
      for act in legalActions:
        stateValue = self.minimax(actedGameState, act, depth - 1, nextAgentIndex, maxValue, beta)
        maxValue = max(maxValue, stateValue)
        if maxValue >= beta:
          break
      return maxValue
    else:
      # Min node
      minValue = beta
      for act in legalActions:
        stateValue = self.minimax(actedGameState, act, depth - 1, nextAgentIndex, alpha, minValue)
        minValue = min(minValue, stateValue)
        if minValue <= alpha:
          break
      return minValue

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()
    #self.badPositions = []
    #self.badPositionRadius = 1
    #self.eatenFood = 0
    #self.entrancePosition = None

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    #last_observation = self.getPreviousObservation()
    #if last_observation and last_observation.getAgentState(self.index).isPacman and not gameState.getAgentState(self.index).isPacman:
    #  print "Virei fantasma"
    #  if self.eatenFood == 0:
    #    self.badPositions.append(self.entrancePosition)
    #    for i in range(1, self.badPositionRadius + 1):
    #      self.badPositions.append((self.entrancePosition[0], self.entrancePosition[1] + 1.0))
    #      self.badPositions.append((self.entrancePosition[0], self.entrancePosition[1] - 1.0))
    #elif last_observation and not last_observation.getAgentState(self.index).isPacman and gameState.getAgentState(self.index).isPacman:
    #  print "Virei pacman"
    #  self.entrancePosition = gameState.getAgentState(self.index).getPosition()
    #  self.eatenFood = 0
    #  self.badPositions = []

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    #print zip(actions, values)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    #actions = gameState.getLegalActions(self.index)
    #values = [self.minimax(gameState, act, 5, self.index, "-inf", "+inf") for act in actions]
    #maxValue = max(values)
    #bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    x = random.choice(bestActions)
    return x
 
class Defender(CaptureAgent):
  "Agente defensivo simples."

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

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    x = random.choice(bestActions)
    return x

  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    #self.target = None

  # Implemente este metodo para controlar o agente (1s max).
  #def chooseAction(self, gameState):
  # alpha = 1 # siga o adversario.
  # mypos = gameState.getAgentPosition(self.index)
  #
  # if mypos == self.target:
  #   self.target = None

  # # Se existir algum Pac-Man no campo de percepcao,
  # # defina target como a posicao do invasor mais proximo.
  # print "====================="
  # x = self.getOpponents(gameState)

  # enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
  # invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
  # if len(invaders) > 0:
  #   positions = [agent.getPosition() for agent in invaders]
  #   self.target = min(positions, key = lambda x: self.getMazeDistance(mypos, x))
  #   if gameState.getAgentState(self.index).scaredTimer > 0:
  #     alpha = -1 # fuja do adversario.

  # # Nenhum inimigo a vista, selecione um pac-dot aleatorio para proteger.
  # if self.target == None:
  #   food = self.getFoodYouAreDefending(gameState).asList() \
  #        + self.getCapsulesYouAreDefending(gameState)
  #   self.target = random.choice(food)

  # # Expande os estados sucessores.
  # # Funcao de avaliacao com base na distancia ao target.
  # actions = gameState.getLegalActions(self.index)
  # fvalues = []
  # for a in actions:
  #   new_state = gameState.generateSuccessor(self.index, a)
  #   newpos = new_state.getAgentPosition(self.index)
  #   fvalues.append(alpha * self.getMazeDistance(newpos, self.target))

  # # Seleciona aleatoriamente entre os estados empatados.
  # best = min(fvalues)
  # ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
  # return random.choice(ties)[1]
