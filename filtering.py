from HiddenMarkovModel import HiddenMarkovModel
from setupAndTraining import TrainingPrebHMM
import numpy as np

class Filtering(HiddenMarkovModel):
    #def __init__(self, emissionProbs, transitionProbs, hiddenStates, observed):
        #HiddenMarkovModel.__init__(emissionProbs, transitionProbs, hiddenStates, observed)

    def filter(self, state, t):
        if (t < 0):
            return 1
        else:
            if self.isComputed(t, state):
                prob = self.getMatrixValue(t,state)
                return prob
            else:
                emissionprob = self.emissionProbability(state, self.observed[t])
                #emmis prob in sum schritt
                sum = 0
                for previousState in self.hiddenStates:
                    sum += self.transitionProbability(previousState, state) * self.filter(previousState, t-1)

                if not sum:
                    #summe wird auf 1 gesetzt
                    sum = 1
                prob = emissionprob * sum
                self.setValueMatrix(t, state, prob)

                return prob


if __name__ == "__main__":

    eingabe = input("Eingabe: ")

    observed = eingabe.split()

    training = TrainingPrebHMM('tags/trainingTags.tags')

    Filter = Filtering(training.emissionProbs, training.transitionProbs, training.hiddenStatesEmission, observed)

    for state in Filter.hiddenStates:
        Filter.filter(state, (len(observed)-1))

    mostLikelyTags = Filter.mostLikelyTags()
    print "Most likely Tags:"
    print mostLikelyTags

