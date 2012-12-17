from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import random

#############
# FACTORIES #
###############################################
# Instanciam os agentes no inicio da partida. #
# Devem estender a classe base AgentFactory.  #
###############################################

class SimpleTeamFactory(AgentFactory):
  "Gera um time SimpleTeam"

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

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  # Implemente este metodo para pre-processamento (15s max).
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.distancer.getMazeDistances()

  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    # Seleciona o pac-dot mais proximo no terreno inimigo.
    mypos  = gameState.getAgentPosition(self.index)
    food   = self.getFood(gameState).asList() + self.getCapsules(gameState)
    target = min(food, key = lambda x: self.getMazeDistance(mypos, x))
      
    # Expande os estados sucessores. 
    # Funcao de avaliacao com base na distancia ao pac-dot.
    actions = gameState.getLegalActions(self.index)
    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      newpos = new_state.getAgentPosition(self.index)
      fvalues.append(self.getMazeDistance(newpos, target))

    # Seleciona aleatoriamente entre os estados empatados.
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    return random.choice(ties)[1]
  
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
    alpha = 1 # siga o adversario.
    mypos = gameState.getAgentPosition(self.index)
    
    if mypos == self.target:
      self.target = None

    # Se existir algum Pac-Man no campo de percepcao,
    # defina target como a posicao do invasor mais proximo.
    enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    if len(invaders) > 0:
      positions = [agent.getPosition() for agent in invaders]
      self.target = min(positions, key = lambda x: self.getMazeDistance(mypos, x))
      if gameState.getAgentState(self.index).scaredTimer > 0:
        alpha = -1 # fuja do adversario.

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
