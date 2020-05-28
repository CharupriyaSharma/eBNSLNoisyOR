import itertools as it 
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import RandomizedSearchCV
from LogRegCPDClass import LogRegCPD
from math import log
from scipy.stats import uniform
from collections import deque
from scipy.special import gamma

def createIndices(parents):
    indices = []
    indices.append(0)
    for d in parents:
        indices.append(0)
    return indices    

def createIndicesParentsOnly(parents):
    indices = []
    for d in parents:
        indices.append(0)
    return indices    

def incrementIndices(indices):
    i = len(indices)-1
    while (i >= 0 and indices[i] == 1):
        indices[i] = 0
        i = i-1
    if i == -1:
        return []
    else:
        indices[i] = indices[i] + 1
        return indices

def calculateLogLikelihoodTerm(cpd, childValue, parentValues):
    noo = cpd.getTable(childValue, parentValues)
    if noo == 0:
        if (cpd.parents == [1,2]):
            print("calculateLogLikelihoodTerm", "Child =", cpd.child, "; Parents =", cpd.parents, 
                  "; Child Value =", childValue, "; Parent Values =", parentValues, 
                  "; Number of Obs =", noo) 
        return 0
    p = cpd.getProb(childValue, parentValues)
    if p == 0:
        if (cpd.parents == [1,2]):
            print("calculateLogLikelihoodTerm", "Child =", cpd.child, "; Parents =", cpd.parents, 
                  "; Child Value =", childValue, "; Parent Values =", parentValues, 
                  "; Number of Obs =", noo,  
                  "; Probability =", p, ) 
        return 100000
    if (cpd.parents == [1,2]):
        print("calculateLogLikelihoodTerm", "Child =", cpd.child, "; Parents =", cpd.parents, 
              "; Child Value =", childValue, "; Parent Values =", parentValues, 
              "; Number of Obs =", noo, "; Probability =", p, 
              "; Result =", noo * log(p))      
    return noo * log(p)  * -1.0

def scoreParentSetLogRegBIC(data, cpd):
    # We make a list that contains a counter for the instantiations of the child node 
    # and each parent node
    indices = createIndices(cpd.parents)
    s = 0
    while (indices != []):
        childValue = indices[0]
        parentValues = indices[1:]      
        s = s + calculateLogLikelihoodTerm(cpd, childValue, parentValues)
        indices = incrementIndices(indices)

    # Computing the BIC penalty term
    numberOfNonZeroCoefficients = len(cpd.coefficients )
    #print(cpd.coefficients)
    #print(numberOfNonZeroCoefficients)
    penalty = (numberOfNonZeroCoefficients+1) * log(data.numberOfRows) / 2
    if (cpd.parents == [1,2]):
        print("scoreParentDataSetLogReg", cpd.child, cpd.parents, s, penalty)
    return s + penalty

def scoreParentSetLogRegBDEU(data, cpd):
    # We make a list that contains a counter for the instantiations of the child node 
    # and each parent node
    indices = createIndicesParentsOnly(cpd.parents)
    s = 0
    while (indices != []):
        noo = cpd.getTable(0, indices) + cpd.getTable(1, indices)
        if noo == 0:
            return 0
        term = log(gamma(1) / gamma(1 + noo))
        term = term + log(gamma(0.5 + cpd.getTable(0, indices)) / gamma(0.5))
        term = term + log(gamma(0.5 + cpd.getTable(1, indices)) / gamma(0.5))
         
        s = s + term
        
        indices = incrementIndices(indices)

    return -s

def scoreParentSetLogReg(data, cpd, scoreName = "BIC"):
    if scoreName == "BIC":
        return scoreParentSetLogRegBIC(data, cpd)
    if scoreName == "BDEU":
        return scoreParentSetLogRegBDEU(data, cpd)
    

class PruneTreeNodeLogReg:
    
    def __init__(self, hashMap, nextQueue, parentTreeNode, data, childNodeIndex, 
                 numberOfVariables, parentSet, score, X, 
                 completePruning, outputfile, outputfile1, scoreName = "BIC"):
        #print("Scored: ", parentSet, score)
        self.scoreName = scoreName
        self.hashMap = hashMap
        self.nextQueue = nextQueue
        self.parentTreeNode = parentTreeNode
        self.data = data
        self.childNodeIndex = childNodeIndex
        self.numberOfVariables = numberOfVariables
        self.parentSet = parentSet
        self.score = score
        self.children = []
        #print("A Call from: ", self.parentSet)
        self.X = X        
        self.completePruning = completePruning
        self.outputfile = outputfile
        self.outputfile1 = outputfile1
        #self.createChildren()
        ###print("2:",hashMap)

        
    def penalty(self,sizeOfSet):
        if self.scoreName == "BIC":
            return (sizeOfSet+1) * log(self.data.numberOfRows) / 2.0 #penalty
        return 0
    
    def __createNumber(self, parentSet):
        result = 0
        for p in parentSet:
            result = result | (1 << p)
        return result    
    
    def __createSet(self, number):
        result = []
        index = 1
        while (number > 0):
            result = result + (number % 2)**index
            number = number / 2
            index = index + 1
        return result
    

        
    def createChildren(self, outputfile, outputfile1):
        ###print("Create children", self.parentSet)
        # Only add variables with a higher index than the current highest index
        # to avoid multiple creations of the same parent set
        
        # Find the highest index in the current set
        maximum = -1
        for i in self.parentSet:
            if i > maximum:
                maximum = i
        
        # Determine the penalty term of children of this node
        childParentSize = len(self.parentSet) + 1
        childPenalty = self.penalty(childParentSize)
        
        # For every potential addition to the current parent set, add the 
        # potential addition to the queue
        for i in range(maximum+1,self.numberOfVariables):             
            if i != self.childNodeIndex:
                if len(self.data.domains[i]) > 1:
                    # Try adding i to the current parent set
                    self.tryCreatingChild(i, childPenalty, outputfile, outputfile1)
                else:
                    #print("Ignoring unary node:", i)
                    fixlater=1
                
    def tryCreatingChild(self, i, childPenalty, outputfile, outputfile1):
        ###print("    Try creating child", i)
        
        # Determine the best score that a subset of the potential parent set has
        bestSubsetScore = 10000
        bestSubset = None            
        
        # Create the child parent set        
        childParentSet = list(self.parentSet)
        childParentSet.append(i)        
                
        # For every potential cardinalty of a true subset
        for j in range(0,len(childParentSet)):
            # Create all subsets of that size
            subsets = list(it.combinations(childParentSet,j))
            ###print("      Subsets: ", j, subsets)
            # For all subsets of that size
            for s in subsets:
                # Convert the set into a bit string
                subsetNumber = self.__createNumber(s)
                ###print("      HashMap: ", subsetNumber, self.hashMap)
                # If the bit string is in the hash map
                if subsetNumber in self.hashMap.keys():
                    ###print(self.hashMap[subsetNumber])
                    # Check whether its score prunes this set
                    #if self.hashMap[subsetNumber] < childPenalty - self.X:
                        #print("  Pruned all children of", self.parentSet)
                    #    return
                    if self.hashMap[subsetNumber] < bestSubsetScore:
                        bestSubsetScore = self.hashMap[subsetNumber]
                        bestSubset = subsetNumber
        ###print("    Creating child", childParentSet, bestSubsetScore, bestSubset)
        self.createChild(childParentSet, bestSubsetScore, bestSubset, outputfile, outputfile1)                
        
    def filterTerms(self, x, featureNames, selection):
        #print('selction', selection)
        #print('names',[featureNames[i] for i in selection])
        return x[:,selection], [featureNames[i] for i in selection] 
        
    def scoreBIC(self, candidateParentSet, x, y, featureNames):

        model = LogisticRegression(solver='liblinear', random_state=0, C=20, max_iter=300, tol=0.000001).fit(x, y)

        nonZeroIndices = np.nonzero(model.coef_)[1]
        nonZeroNames = [featureNames[i].replace('x','').split(' ') for i in nonZeroIndices]
        #print(nonZeroIndices)
        #print(len(nonZeroIndices)) 
        new_list = [[int(x) for x in lst] for lst in nonZeroNames]
        if (candidateParentSet == [1]):
            print("scoreBIC", model.coef_,model.coef_[0], np.nonzero(model.coef_)[1], new_list, nonZeroIndices)   
        if (len(candidateParentSet) == 1):
            logRegCPD = LogRegCPD(self.data, model.intercept_, new_list, model.coef_[0], np.nonzero(model.coef_)[1],
                              self.childNodeIndex,
                              candidateParentSet)    
        else:
            logRegCPD = LogRegCPD(self.data, model.intercept_, new_list, [num for num in model.coef_[0] if num], nonZeroIndices, 
                              self.childNodeIndex,
                              candidateParentSet)

        bic = scoreParentSetLogReg(self.data, logRegCPD, self.scoreName)
        #print("PruneTreeNodeClass.py", "PruneTreeNodeLogReg.scoreBIC", "parentSet=", candidateParentSet,
        #      "child=" , self.childNodeIndex, bic)
        return bic, model, nonZeroIndices
        
    def initialSelections(self, selectionsQueue, childParentSet):
        selectionsQueue.append(list(range(0,len(childParentSet))))
    
    def expand(self, selectionsQueue, currentSelection, expansionsCounter, featureNames):
        if expansionsCounter > 1000:
            return expansionsCounter
        maxLen = len(featureNames)
        maxEl = 0
        for i in currentSelection:
            if maxEl < i:
                maxEl = i
        for i in range(1,10):
            if maxEl + i < maxLen:
                newSelection = currentSelection.copy()            
                newSelection.append(maxEl+i)
                selectionsQueue.append(newSelection)
        return expansionsCounter+10         
            
    
    def createChild(self, childParentSet, bestScoreAbove, bestSubsetAbove, outputfile, outputfile1):
        ###print("Create child", i)
        fo = open(outputfile,"a+")
        fo1 = open(outputfile1,"a+")

        x = self.data.data[:,childParentSet]
        y = self.data.data[:,self.childNodeIndex]
        #print(x)
        poly = PolynomialFeatures(interaction_only=True,include_bias = False, degree=len(childParentSet)) 
        x=poly.fit_transform(x)
        featureNames=poly.get_feature_names()
        selections = deque()
        self.initialSelections(selections, childParentSet)
        currentBestSelections = []

        expansionsCounter = 0
        bestNonZeroIndices =-1
        
        bestBIC = 100000
        
        #for selection in selections:
        while selections:    
            selection = selections.popleft()
            #print(selections)
#            if selection not in [L[1] for L in currentBestSelections]:
#                continue
            xF, featureNamesF = self.filterTerms(x, featureNames, selection) 
            #print(xF)
            #print(xF.shape)

            bic, model, nonZeroIndices = self.scoreBIC(childParentSet, xF, y, featureNamesF)
            currentBestSelections.append([bic, selection])
            currentBestSelections.sort()
            if len(currentBestSelections) > 15:
                currentBestSelections.pop()
            if bic in [L[0] for L in currentBestSelections]:
                bestBIC = bic
                bestModel = model
                bestNonZeroIndices = nonZeroIndices
               
                expansionsCounter = self.expand(selections, selection, expansionsCounter, featureNames)
        
        
        #selections = deque() #Queue()# = list(range(0,len(childParentSet)))
        #self.initialSelections(selections, childParentSet)
        #expansionsCounter = 0
        
        #bestBIC = 100000        
        ##for selection in selections:
        #while selections:    
        #    selection = selections.popleft()
        #    xF, featureNamesF = self.filterTerms(x, featureNames, selection)        
        #    bic, model, nonZeroIndices = self.scoreBIC(childParentSet, xF, y, featureNamesF)
        #    if bic < bestBIC:
        #    #    bestBIC = bic
        #        bestModel = model
        #        bestNonZeroIndices = nonZeroIndices
        #        expansionsCounter = self.expand(selections, selection, expansionsCounter, featureNames)
        

        
        #model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
        
        #nonZeroIndices = np.nonzero(model.coef_)[1]
        #nonZeroNames = [poly.get_feature_names()[i].replace('x','').split(' ') for i in nonZeroIndices]
        
        #new_list = [[int(x) for x in lst] for lst in nonZeroNames]
        #logRegCPD = LogRegCPD(self.data, model.intercept_, new_list, nonZeroIndices, self.childNodeIndex,
        #                      childParentSet)

        #bic = scoreParentSetLogReg(self.data, logRegCPD)
        if (bic < bestScoreAbove + self.X):
            #print("Adding:", childParentSet, bic)
            self.hashMap[self.__createNumber(childParentSet)] = bic
            output = "   " + str(-bic) + " " + str(len(childParentSet))
            output1 = str(len(bestNonZeroIndices)) + " " + str(len(childParentSet))
            #print(xF)
            #print(output1)
            for n in childParentSet:
                output = output + " " + str(n)
                output1 = output1 + " " + str(n)

   #         for n in childParentSet:
    #            output1 = output1 + " " + str(noiseParams[n])

            fo.write(output + "\n")
            fo1.write(output1 + "\n") 
        child = PruneTreeNodeLogReg(self.hashMap, self.nextQueue, self, self.data, self.childNodeIndex, self.numberOfVariables, 
                              childParentSet, bic, self.X, self.completePruning, self.outputfile, 
                              self.outputfile1, self.scoreName)
        self.children.append(child) 
        self.nextQueue.append(child)        
        ###print("Created", i)
        fo.close()
        fo1.close()
