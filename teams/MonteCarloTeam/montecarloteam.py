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

class MonteCarloFactory(AgentFactory):
  "Gera um time MonteCarloTeam"

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

class EvaluationBasedAgent(CaptureAgent):
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
    #print action
    #print features
    #print weights
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}


class Attacker(EvaluationBasedAgent):
  "Agente ofensivo simples."

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # Compute score from successor state
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute distance to closest ghost
    myPos = successor.getAgentState(self.index).getPosition()
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(inRange) > 0:
      positions = [agent.getPosition() for agent in inRange]
      closest = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      closestDist = self.getMazeDistance(myPos, closest)
      if closestDist <= 5:
        features['distanceToGhost'] = closestDist

    # Compute distance to ally
    team = self.getTeam(successor)
    team.remove(self.index)
    ally = successor.getAgentState(team[0])
    allyPosition = ally.getPosition()
    allyDist = self.getMazeDistance(myPos, allyPosition)
    features['distanceToAlly'] = allyDist

    # Compute walked distance
    features['walkedDist'] = self.getMazeDistance(myPos, self.lastPosition)

    # Compute if is pacman
    features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

    return features

  def getWeights(self, gameState, action):
    # If tha agent is locked, we will make him try and atack
    if self.inactiveTime > 80:
      return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'distanceToAlly': 0, 'walkedDist': 1, 'isPacman': 1000}

    # If opponent is scared, the agent should not care about distanceToGhost
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(inRange) > 0:
      positions = [agent.getPosition() for agent in inRange]
      closestPos = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      closestDist = self.getMazeDistance(myPos, closestPos)
      closest_enemies = filter(lambda x: x[0] == closestPos, zip(positions, inRange))
      for agent in closest_enemies:
        if agent[1].scaredTimer > 0:
          return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'distanceToAlly': 0, 'walkedDist': 1, 'isPacman': 0}

    # Weights normally used
    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'distanceToAlly': 0, 'walkedDist': 1, 'isPacman': 0}

  def randomSimulation(self, depth, gameState):
    new_state = gameState.deepCopy()
    while depth > 0:
      # Get valid actions
      actions = new_state.getLegalActions(self.index)
      # The agent should not stay put in the simulation
      actions.remove(Directions.STOP)
      current_direction = new_state.getAgentState(self.index).configuration.direction
      # The agent should not use the reverse direction during simulation
      reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
      if reversed_direction in actions and len(actions) > 1:
        actions.remove(reversed_direction)
      # Randomly chooses a valid action
      a = random.choice(actions)
      # Compute new state and update depth
      new_state = new_state.generateSuccessor(self.index, a)
      depth -= 1
    # Evaluate the final simulation state
    return self.evaluate(new_state, Directions.STOP)

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()
    # Variables used to verify if the agent os locked
    self.numEnemyFood = "+inf"
    self.inactiveTime = 0

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    # You can profile your evaluation time by uncommenting these lines
    #start = time.time()

    # Stores the agent position. This value will be used in the state evaluation.
    self.lastPosition = gameState.getAgentState(self.index).getPosition()

    # Updates inactiveTime. This variable indicates if the agent is locked.
    currentEnemyFood = len(self.getFood(gameState).asList())
    if self.numEnemyFood != currentEnemyFood:
      self.numEnemyFood = currentEnemyFood
      self.inactiveTime = 0
    else:
      self.inactiveTime += 1
    # If the agent dies, inactiveTime is reseted.
    if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
      self.inactiveTime = 0

    # Get valid actions. Staying put is almost never a good choice, so
    # the agent will ignore this action.
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)

    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      value = 0
      for i in range(1,31):
        value += self.randomSimulation(6, new_state)
      fvalues.append(value)

    best = max(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    toPlay = random.choice(ties)[1]

    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    return toPlay
 
class Defender(CaptureAgent):
  "Agente defensivo simples."

  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.target = None
    self.lastObservedFood = None

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):

    alpha = 1 # siga o adversario.
    mypos = gameState.getAgentPosition(self.index)
   
    if mypos == self.target:
      self.target = None

    # Se existir algum Pac-Man no campo de percepcao,
    # defina target como a posicao do invasor mais proximo.
    x = self.getOpponents(gameState)

    enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    if len(invaders) > 0:
      positions = [agent.getPosition() for agent in invaders]
      self.target = min(positions, key = lambda x: self.getMazeDistance(mypos, x))
      if gameState.getAgentState(self.index).scaredTimer > 0:
        alpha = -1 # fuja do adversario.
    # Se nao existir um pacman no campo de percepcao,
    # mas nossos pacdots estiverem sumindo, va na direcao
    # dos pacdots desaparecidos
    elif self.lastObservedFood != None:
      eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
      if len(eaten) > 0:
        self.target = eaten.pop()

    self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

    # Nenhum inimigo a vista, selecione um pac-dot aleatorio para proteger.
    if self.target == None:
      food = self.getFoodYouAreDefending(gameState).asList() \
           + self.getCapsulesYouAreDefending(gameState)
      self.target = random.choice(food)

    # Expande os estados sucessores.
    # Funcao de avaliacao com base na distancia ao target.
    actions = gameState.getLegalActions(self.index)
    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      newpos = new_state.getAgentPosition(self.index)
      fvalues.append(alpha * self.getMazeDistance(newpos, self.target))

    # Seleciona aleatoriamente entre os estados empatados.
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    return random.choice(ties)[1]
