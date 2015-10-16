# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    self.qValues = util.Counter() #Key: (state, action) pair, Value: q-value

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    return self.qValues[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    return max([self.getQValue(state, action) for action in self.getLegalActions(state)])

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    STATE, ACTION, Q_VALUE = 0, 1, 2
    if len(self.getLegalActions(state)) == 0:
        return None
    qValuesForState = [(state, action, self.getQValue(state, action)) for action in self.getLegalActions(state)]
    #print "qValuesForState:", qValuesForState
    #print "max of qValuesForState:", max(qValuesForState, key=lambda x: x[Q_VALUE])
    maxQValues = [stateActionQValue for stateActionQValue in qValuesForState
                  if stateActionQValue[Q_VALUE] == max(qValuesForState, key=lambda x: x[Q_VALUE])[Q_VALUE]]
    #print "maxQValues:", maxQValues
    return maxQValues[0][ACTION] if len(maxQValues) == 1 else random.choice(maxQValues)[ACTION]

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    if len(legalActions) == 0:
        #print "No legal actions"
        action = None
    elif util.flipCoin(self.epsilon):
        #print "Random Choice of Action"
        action = random.choice(legalActions)
    else:
        #print "Choice of action based on Policy"
        action = self.getPolicy(state)
    #print "Action:", action
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    nextStateQValues = [self.qValues[(nextState, nextAction)] for nextAction in self.getLegalActions(nextState)]
    sample = reward + self.discount*(max(nextStateQValues) if len(nextStateQValues) > 0 else 0)
    self.qValues[(state, action)] = (1 - self.alpha)*self.qValues[(state, action)] + self.alpha*sample

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    if self.weights[(state, action)] == 0:
        self.weights[(state, action)] = util.Counter()
        return 0
    else: return self.weights[(state, action)] * self.featExtractor.getFeatures(state, action)

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    qValue = self.getQValue(state, action)
    nextStateQValues = [self.qValues[(nextState, nextAction)] for nextAction in self.getLegalActions(nextState)]
    correction = (reward + self.discount*(max(nextStateQValues) if len(nextStateQValues) > 0 else 0)) - qValue
    print "Correction:", correction
    if self.weights[(state, action)] == 0:
        self.weights[(state, action)] = util.Counter()
    else:
        features = self.featExtractor.getFeatures(state, action)
        print "alpha:", self.alpha
        if self.alpha*correction == 0:
            features[(state, action)] = 0
        else:
            features.divideAll(1./(self.alpha*correction))
        self.weights[(state, action)] += features


  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      print "Final Weights:", self.weights.values()
