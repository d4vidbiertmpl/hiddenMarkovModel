from __future__ import division
import numpy as np
from itertools import groupby


class HiddenMarkovModel():
    def __init__(self, emissionProbs, transitionProbs, hiddenStates, observed):

        self.hiddenStates = hiddenStates
        self.transitionProbabilities = transitionProbs
        self.emissionProbabilities = emissionProbs
        self.initProbabilities = {}
        self.initProbabilities = self.initializeInitProbabilities()

        self.hiddenStates = hiddenStates

        self.observed = observed

        self.probabilityMatrix = np.zeros(shape=(len(self.observed), len(self.hiddenStates)))

        self.computedMatrix = np.zeros(shape=(len(self.observed), len(self.hiddenStates)))




    def initializeInitProbabilities(self):
        for state in self.hiddenStates:
            if not ('SatzAnfang',state) in self.transitionProbabilities:
                self.transitionProbabilities[('SatzAnfang', state)] = 0.0

        for state in self.hiddenStates:
            if state in self.initProbabilities:
                self.initProbabilities[state].append(self.transitionProbabilities[('SatzAnfang', state)])
            else:
                self.initProbabilities[state] = [self.transitionProbabilities[('SatzAnfang', state)]]



    def setValueMatrix(self, t, state, value):
        self.setToComputed(t, state)
        self.probabilityMatrix[t][self.getStatePosition(state)] = value


    def getMatrixValue(self, t, state):
        return self.probabilityMatrix[t][self.getStatePosition(state)]


    def getStatePosition(self, state):
        return self.hiddenStates.index(state)


    def setToComputed(self, t, state):
        self.computedMatrix[t][self.getStatePosition(state)] = 1


    def isComputed(self, t, state):
        return self.computedMatrix[t][self.getStatePosition(state)] == 1


    def transitionProbability(self, state_1, state_2):
        if (state_1, state_2) in self.transitionProbabilities:
            return self.transitionProbabilities[(state_1, state_2)]
        else:
            return 0


    def emissionProbability(self, state, obs):
        if (state, obs) in self.emissionProbabilities:
            return self.emissionProbabilities[(state, obs)]
        else:
            return 0

    def mostLikelyTags(self):
        maxPositions = self.probabilityMatrix.argmax(1)
        tags = []
        for i in maxPositions:
            tag = self.hiddenStates[i]
            if tag == '$.' and self.getMatrixValue(i,tag) == 0:  # not a full spot but an unknown word
                tag = 'NE'  # replace unknown words with NE (proper Noun) increases accuracy considerably (Eugene Charniak, Statistical techniques for natural language parsing (1997))
            tags.append(tag)
        return tags
