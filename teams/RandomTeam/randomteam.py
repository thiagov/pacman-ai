from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import random

#############
# FACTORIES #
###############################################
# Instanciam os agentes no inicio da partida. #
# Devem estender a classe base AgentFactory.  #
###############################################

class RandomTeamFactory(AgentFactory):
  "Gera um time RandomTeam"

  def __init__(self, isRed):
    AgentFactory.__init__(self, isRed)

  def getAgent(self, index):                 
    return RandomAgent(index)

#############
#  AGENTS   #
###############################################
# Implementacoes dos agentes.                 #
# Devem estender a classe base CaptureAgent.  #
###############################################

class RandomAgent(CaptureAgent):
  "Agente aleatorio simples."

  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def chooseAction(self, gameState):    
    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)
