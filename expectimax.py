def cutoffTest(gameState, depth):
  if depth == 0 or gameState.isOver():
    return true
  return false

def minimax(self, gameState, action, depth, agentIndex, alpha, beta):
  if cutoffTest(gameState, depth):
    return evaluate(gameState, action)

  # Perform the given action
  actedGameState = gameState.generateSuccessor(agentIndex, action)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  legalActions = actedGameState.getLegalActions(nextAgentIndex)

  # If the next agent is very far away, it's not worth the extra time to consider his actions; just assume he won't move
  if self.distancer.getDistance(gameState.getAgentPosition(agentIndex), gameState.getAgentPosition(nextAgentIndex)) > (depth * 3):
    self.log(4, 'agent %d is too far away' % (nextAgentIndex))
    legalActions = [Directions.STOP]

  if gameState.isOnRedTeam(agentIndex) == gameState.isOnRedTeam(self.index):
    # Max node
    maxValue = alpha
    for act in legalActions:
      stateValue = self.evaluateMinimax(actedGameState, act, depth - 1, nextAgentIndex, maxValue, beta)
      maxValue = max(maxValue, stateValue)
      if maxValue >= beta:
        self.log(4, 'truncate at plysLeft %d due to beta %s' % (depth, beta))
        break
    return maxValue
  else:
    # Min node
    minValue = beta
    for act in legalActions:
      stateValue = self.evaluateMinimax(actedGameState, act, depth - 1, nextAgentIndex, alpha, minValue)
      minValue = min(minValue, stateValue)
      if minValue <= alpha:
        self.log(4, 'truncate at plysLeft %d due to alpha %s' % (depth, alpha))
        break
    return minValue
