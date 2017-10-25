#!/usr/bin/env python
""" generated source for module BayesianNetwork """
from Assignment4 import *
import random
# 
#  * A bayesian network
#  * @author Panqu
#  
class BayesianNetwork(object):
    """ generated source for class BayesianNetwork """
    # 
    #     * Mapping of random variables to nodes in the network
    #     
    varMap = None

    # 
    #     * Edges in this network
    #     
    edges = None

    # 
    #     * Nodes in the network with no parents
    #     
    rootNodes = None

    # 
    #     * Default constructor initializes empty network
    #     
    def __init__(self):
        """ generated source for method __init__ """
        self.varMap = {}
        self.edges = []
        self.rootNodes = []

    # 
    #     * Add a random variable to this network
    #     * @param variable Variable to add
    #     
    def addVariable(self, variable):
        """ generated source for method addVariable """
        node = Node(variable)
        self.varMap[variable]=node
        self.rootNodes.append(node)

    # 
    #     * Add a new edge between two random variables already in this network
    #     * @param cause Parent/source node
    #     * @param effect Child/destination node
    #     
    def addEdge(self, cause, effect):
        """ generated source for method addEdge """
        source = self.varMap.get(cause)
        dest = self.varMap.get(effect)
        self.edges.append(Edge(source, dest))
        source.addChild(dest)
        dest.addParent(source)
        if dest in self.rootNodes:
            self.rootNodes.remove(dest)

    # 
    #     * Sets the CPT variable in the bayesian network (probability of
    #     * this variable given its parents)
    #     * @param variable Variable whose CPT we are setting
    #     * @param probabilities List of probabilities P(V=true|P1,P2...), that must be ordered as follows.
    #       Write out the cpt by hand, with each column representing one of the parents (in alphabetical order).
    #       Then assign these parent variables true/false based on the following order: ...tt, ...tf, ...ft, ...ff.
    #       The assignments in the right most column, P(V=true|P1,P2,...), will be the values you should pass in as probabilities here.
    #     
    def setProbabilities(self, variable, probabilities):
        """ generated source for method setProbabilities """
        probList = []
        for probability in probabilities:
            probList.append(probability)
        self.varMap.get(variable).setProbabilities(probList)

    def normalize(self, toReturn):
        SUM = toReturn[0] + toReturn[1]

        if SUM is 0:
            return 0, 0
        else:
            return float(toReturn[0])/SUM

    # 
    #     * Returns an estimate of P(queryVal=true|givenVars) using rejection sampling
    #     * @param queryVar Query variable in probability query
    #     * @param givenVars A list of assignments to variables that represent our given evidence variables
    #     * @param numSamples Number of rejection samples to perform
    #     
    def performRejectionSampling(self, queryVar, givenVars, numSamples):
        """ generated source for method performRejectionSampling """
        #  TODO
        toReturn = [0, 0]
        for j in range(1, numSamples):
            # start prior sampling
            x = {}
            sortVar = sorted(self.varMap)
            for variable in sortVar:
                ran = random.random()
                if ran <= self.varMap[variable].getProbability(x, True):
                    x[variable.getName()] = True
                else:
                    x[variable.getName()] = False
            # end prior sampling

            for e in givenVars:
                if x[e.getName()] == givenVars[e]:
                    if x[queryVar.getName()] is True:
                        toReturn[0] += 1
                    else:
                        toReturn[1] += 1

        return self.normalize(toReturn)

    # 
    #     * Returns an estimate of P(queryVal=true|givenVars) using weighted sampling
    #     * @param queryVar Query variable in probability query
    #     * @param givenVars A list of assignments to variables that represent our given evidence variables
    #     * @param numSamples Number of weighted samples to perform
    #     
    def performWeightedSampling(self, queryVar, givenVars, numSamples):
        """ generated source for method performWeightedSampling """
        #  TODO
        toReturn = [0, 0]
        for j in range(1, numSamples):
            # weightedSample
            (x, w) = self.weightedSample(self.varMap, givenVars)

            if x[queryVar.getName()] is True:
                toReturn[0] += w
            else:
                toReturn[1] += w

        return self.normalize(toReturn)

    def weightedSample(self, bn, e):
        x = Sample()
        for event in e:
            x.setAssignment(event.getName(), e[event])

        sortVar = sorted(bn.keys())
        for xi in sortVar:
            if x.getValue(xi.getName()) is not None:
                w = x.getWeight()
                w = w * bn[xi].getProbability(x.assignments, x.assignments.get(xi.getName()))
                x.setWeight(w)
            else:
                ran = random.random()
                if ran <= bn[xi].getProbability(x.assignments, True):
                    x.assignments[xi.getName()] = True
                else:
                    x.assignments[xi.getName()] = False

        return x.assignments, x.getWeight()


    #
    #     * Returns an estimate of P(queryVal=true|givenVars) using Gibbs sampling
    #     * @param queryVar Query variable in probability query
    #     * @param givenVars A list of assignments to variables that represent our given evidence variables
    #     * @param numTrials Number of Gibbs trials to perform, where a single trial consists of assignments to ALL
    #       non-evidence variables (ie. not a single state change, but a state change of all non-evidence variables)
    #     
    def performGibbsSampling(self, queryVar, givenVars, numTrials):
        """ generated source for method performGibbsSampling """
        #  TODO
        counter = [0, 0]

        nonEviVar = []
        givenVarsSort = sorted(givenVars)

        newvarMap = {}

        # set all needed variable field
        for variable in self.varMap.keys():
            if variable in givenVarsSort:
                newvarMap[variable.getName()] = givenVars[variable]
                continue
            else:
                nonEviVar.append(variable)
                randomprob = random.random()
                if randomprob < 0.5:
                    newvarMap[variable.getName()] = False
                else:
                    newvarMap[variable.getName()] = True

        # gibbs sampling
        # idea from book page 537
        for j in range(1, numTrials):
            for z in nonEviVar:
                markovList = self.markovBlanket(self.varMap.get(z))
                markovMap = {}

                for mark in markovList:
                    markovMap[mark.getVariable().getName()] = newvarMap[mark.getVariable().getName()]
                probCom = self.gibbsProb(markovMap, z)
                if probCom[0] is 0:
                    alpha = 0
                else:
                    alpha = 1.0/probCom[0]
                val = alpha * probCom[1]

                randomprob2 = random.random()
                if val < randomprob2:
                    newvarMap[self.varMap[z].getVariable().getName()] = False
                else:
                    newvarMap[self.varMap[z].getVariable().getName()] = True

                if newvarMap[queryVar.getName()] is False:
                    counter[1] += 1
                else:
                    counter[0] += 1

        return self.normalize(counter)


    def gibbsProb(self, markMap, Z_i):
        query = {}
        probC_true = 1.0
        probC_false = 1.0

        for par in self.varMap[Z_i].getParents():
            query[par.getVariable().getName()] = markMap[par.getVariable().getName()]

        prob_true = self.varMap[Z_i].getProbability(query, True)
        prob_false = self.varMap[Z_i].getProbability(query, False)

        for child in self.varMap[Z_i].getChildren():
            childP = {}
            for childp in child.getParents():
                if childp.getVariable().equals(self.varMap[Z_i].getVariable()) is False:
                    childP[childp.getVariable().getName()] = markMap[childp.getVariable().getName()]
                else:
                    childP[childp.getVariable().getName()] = True

            probC_true = prob_true * child.getProbability(childP, markMap[child.getVariable().getName()])
        for child in self.varMap[Z_i].getChildren():
            childP = {}
            for childp in child.getParents():
                if childp.getVariable().equals(self.varMap[Z_i].getVariable()) is False:
                    childP[childp.getVariable().getName()] = markMap[childp.getVariable().getName()]
                else:
                    childP[childp.getVariable().getName()] = False

            probC_false = prob_false * child.getProbability(childP, markMap[child.getVariable().getName()])
        toReturn = prob_true * probC_true + prob_false * probC_false
        return toReturn, prob_true * probC_true



    # markovBlanket method that provides a list that we need to use for Gibbs Sampling
    # idea from slide 19_20 page 18
    def markovBlanket(self, node):
        markovList = []
        for parentN in node.getParents():
            markovList.append(parentN)

        for childrenN in node.getChildren():
            markovList.append(childrenN)

            for parentC in childrenN.getParents():
                if parentC is node or parentC in markovList:
                    continue
                markovList.append(parentC)

        return markovList

