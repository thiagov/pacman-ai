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
      features['distanceToGhost'] = self.getMazeDistance(myPos, closest)

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
      min_dist = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
      closest_enemies = filter(lambda x: x[0] == min_dist, zip(positions, in_range))
      for agent in closest_enemies:
        if agent[1].scaredTimer > 0:
          # If opponent is scared, the agent should not care about distanceToGhost.
          return {'successorScore': 3, 'distanceToFood': -1, 'distanceToCapsule': -0.5, 'distanceToGhost': 0, 'distanceToAlly': 2}
      # If agent is being persued, the agent should focus on surviving: it should
      # give priority to distanceToCapsule and distanceToGhost
      return {'successorScore': 3, 'distanceToFood': -1, 'distanceToCapsule': -5, 'distanceToGhost': 5, 'distanceToAlly': 2}
    else:
      # Normal weights
      return {'successorScore': 3, 'distanceToFood': -1, 'distanceToCapsule': -0.5, 'distanceToGhost': 1, 'distanceToAlly': 2}

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
    print features
    return features * weights

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()
    #print gameState.data.layout
    #print  dir(gameState)
    #print gameState.getWalls()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)
 
class Defender(CaptureAgent):
  "Agente defensivo simples."

  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.target = None

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    return "Stop"
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
