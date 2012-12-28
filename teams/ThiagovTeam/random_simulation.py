from captureAgents import CaptureAgent
from captureAgents import AgentFactory
from game import Directions
import random

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

  def randomSimulation(self, depth, gameState):
    new_state = gameState.deepCopy()
    while depth > 0:
      #pega jogadas validas
      actions = new_state.getLegalActions(self.index)
      #nao queremos o agente parado na simulacao
      actions.remove(Directions.STOP)
      current_direction = new_state.getAgentState(self.index).configuration.direction
      #nao queremos o agente indo e voltando na simulacao
      reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
      if reversed_direction in actions and len(actions) > 1:
        actions.remove(reversed_direction)
      #escolhe a acao aleatoriamente
      a = random.choice(actions)
      #print new_state
      #print actions
      new_state = new_state.generateSuccessor(self.index, a)
      depth -= 1
    #calcula e retorna valor da jogada
    return self.getScore(new_state)

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()
    print  dir(gameState)
    print gameState.getWalls()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    food   = self.getFood(gameState).asList() + self.getCapsules(gameState)
    capsules = self.getCapsules(gameState)
    enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    actions = gameState.getLegalActions(self.index)
    #last_observation = self.getPreviousObservation()
    #if last_observation:
    #  reversed_direction = Directions.REVERSE[last_observation.getAgentState(self.index).configuration.direction]
    #  if reversed_direction in actions and len(actions) > 1:
    #    actions.remove(reversed_direction)

    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      value = 0
      for i in range(1,21):
        value += self.randomSimulation(10, new_state)
      fvalues.append(value)
    #print fvalues
    #print current_direction
    #print reversed_direction
    #return 'Stop'

    best = max(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    return random.choice(ties)[1]
  # # Seleciona o pac-dot mais proximo no terreno inimigo.
  # mypos  = gameState.getAgentPosition(self.index)
  # food   = self.getFood(gameState).asList() + self.getCapsules(gameState)
  # target = min(food, key = lambda x: self.getMazeDistance(mypos, x))
  #
  # # Expande os estados sucessores.
  # # Funcao de avaliacao com base na distancia ao pac-dot.
  # actions = gameState.getLegalActions(self.index)
  # fvalues = []
  # for a in actions:
  #   new_state = gameState.generateSuccessor(self.index, a)
  #   newpos = new_state.getAgentPosition(self.index)
  #   fvalues.append(self.getMazeDistance(newpos, target))

  # # Seleciona aleatoriamente entre os estados empatados.
  # best = min(fvalues)
  # ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
  # return random.choice(ties)[1]
  
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
  # print gameState.getAgentState(x[1])

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
