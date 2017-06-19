from __future__ import division
from itertools import groupby
import numpy as np
from HiddenMarkovModel import HiddenMarkovModel

#Calculation of Emission Probablilities over Emission tuples
#Calculation of transition Probablities over Sentences

class TrainingPrebHMM():
    def __init__(self, file):

        data = self.readWordFile(file)

        emissionDictionary = self.setUpEmissionData(data)
        transitionDictionary = self.setUpTransitionData(data)

        self.hiddenStatesEmission = []

        self.hiddenStatesTransition = []

        self.setUpHiddenStates(emissionDictionary, transitionDictionary)

        self.emissionProbs = self.training(emissionDictionary, self.hiddenStatesEmission)

        self.transitionProbs = self.training(transitionDictionary, self.hiddenStatesTransition)



    def readWordFile(self, file):

        with open(file) as f:
            data = f.read().splitlines()

        # Divide the text data into sentences by grouping the data.
        self.sentences = [list(group) for k, group in groupby(data, lambda x: x == "") if not k]

        # Deleting the spaces in the data
        data = filter(lambda a: a != "", data)

        return data


    def setUpEmissionData(self, data):

        emissionTupelsList = []
        # Split the Data and create (Word,Tag) Tuples
        for i in range(len(data)):
            split = data[i].split('\t')
            tuple = (split[0], split[1])
            emissionTupelsList.append(tuple)

        emissionTupels = np.asarray(emissionTupelsList)


        # Dictionary for calculating the emission probabilities
        emissionDictionary= {}
        for tuple in emissionTupels:
            key = tuple[1]

            if key in emissionDictionary:
                emissionDictionary[key].append(tuple[0])
            else:
                emissionDictionary[key] = [tuple[0]]

        return emissionDictionary


    def setUpTransitionData(self, data):

        # Divide Sentences in sublists
        sentences = []
        for sentence in self.sentences:
            sen = []
            for i in range(len(sentence)):
                split = sentence[i].split('\t')
                sen.append(split)

            sentences.append(sen)

        #Setting up the transition dictionary
        transitionDictionary = {}
        for s in sentences:
            for i in range(len(s)):
                if (i == 0):
                    tag = "SStart"
                    tag_1 = s[i][1]
                elif (i == (len(s)) - 1):
                    tag = s[i][1]
                    tag_1 = "SEnd"
                else:
                    tag = s[i][1]
                    i += 1
                    tag_1 = s[i][1]

                if tag in transitionDictionary:
                    transitionDictionary[tag].append(tag_1)
                else:
                    transitionDictionary[tag] = [tag_1]

        return transitionDictionary


    def setUpHiddenStates(self, emissionDictionary, transitionDictionary):
        # Hidden States
        for key in emissionDictionary:
            self.hiddenStatesEmission.append(key)

        for key in transitionDictionary:
            self.hiddenStatesTransition.append(key)




    #Calculation of Emission/Transition probabilitites. Each PartofSpeech Tag (HiddenStates) gets looked. A tuple is created for all tag word combinations.
    #This tuple is added to the probabilities dictionary with it corresponding value.
    def training(self, dictionary, hiddenStates):

        probabilities = {}

        for i in range(len(hiddenStates)):
            tag = hiddenStates[i]
            for j in range(len(dictionary[tag])):
                word = dictionary[tag][j]
                key = (tag, word)
                if not key in probabilities:
                    wordCount = dictionary[tag].count(key[1])
                    wordPartCount = (len(dictionary[tag]))
                    prob = wordCount / wordPartCount
                    probabilities[key] = prob

        return probabilities
