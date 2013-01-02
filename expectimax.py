def evaluate(gameState):
  return 10

def cutoffTest(gameState, depth):
  if depth == 0:
    return true
  return false

def expectiMinimax(player, self, gameState, depth):
  if cutoffTest(gameState, depth):
    return evaluate(gameState)
  if player == "player":
    alpha = float("-inf")
    actions = gameState.getLegalActions(self.index)
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      max(alpha, expectiMinimax("chance", self, new_state, depth-1))
  elif player == "chance":
    alpha = 0
    opponents = self.getOpponents(gameState)
