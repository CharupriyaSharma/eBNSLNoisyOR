import sys
import itertools as it 
import numpy as np
from math import log, ceil
from collections import deque
from numpy import genfromtxt


class Data:

    # Returns the domain of a node by checking all values it takes from a numpy array 
    def domainOf(self, node):
        domain = []
        for row in self.data:
            if row[node] not in domain:
                domain.append(row[node])
        return domain    

    def domainsOf(self):
        domains = []
        for node in range(0,len(self.data[0])):
            domains.append(self.domainOf(node))
        return domains  
    
    # Reads integer data from a CSV file with delimiter ','.
    def readFromCSVFile(self, filename):
        self.data = genfromtxt(filename, delimiter=',', dtype=int)    
        self.domains = self.domainsOf()
        self.numberOfColumns = len(self.data[0])
        self.numberOfRows = len(self.data)
        
    def computeCPD(self, childNode, parentSet):
        result = CPD(self, childNode, parentSet)        
        return result        

class CPD:
    
    def __init__(self, data, childNode, parentSet):
        self.data = data
        self.child = childNode
        self.parents = parentSet
        self.dimensions = []
        self.dimensions.append(len(data.domains[childNode]))
        for parentNode in parentSet:
            self.dimensions.append(len(data.domains[parentNode]))            
        self.table = np.ndarray(shape=self.dimensions)
        self.table.fill(0)
        self.prob = np.ndarray(shape=self.dimensions)
        self.prob.fill(0)
        
        for row in data.data:
            self.__add(row, childNode, parentSet)
            
        self.__computeProbabilities()             

        
    def __add(self, row, childNode, parentSet):
        if parentSet == []:
            self.table[row[childNode]] = self.table[row[childNode]] + 1
        else:   
            self.__addTo(self.table[row[childNode]], row, parentSet, 0)
        
    def __addTo(self, a, row, parentSet, index):
        if index == len(parentSet)-1:            
            a[row[parentSet[index]]] = a[row[parentSet[index]]] + 1
        else:
            self.__addTo(a[row[parentSet[index]]], row, parentSet, index+1)
    
    def __createParentSetIndices(self):
        indices = []
        for d in self.parents:
            indices.append(0)
        return indices    

    def __incrementParentSetIndices(self, indices):
       i = len(indices)-1
       while (i >= 0 and indices[i] == len(self.data.domains[self.parents[i]])-1):
           indices[i] = 0
           i = i-1
       if i == -1:
           return []
       else:
           indices[i] = indices[i] + 1
           return indices
        
    # Computes the CPT
    def __computeProbabilities(self):
        # Create indices, starting with the all 0 indices
        if self.parents == []:
            observationCount = 0
            for childValue in self.data.domains[self.child]:
                observationCount = observationCount + self.getTable(childValue,[])
            # Use the count to transform the counts into conditional 
            # probabilities                
            if observationCount == 0:
                self.setProb(childValue, [], 0)
                #print(type(self.getTable(childValue,indices)))
            else:
                for childValue in self.data.domains[self.child]:
                    self.setProb(childValue, [], self.getTable(childValue,[]) / observationCount)
            return
        indices = self.__createParentSetIndices()
        while (indices != []):
            # Counts all observations that fit the current realization of the 
            # parent set
            observationCount = 0
            for childValue in self.data.domains[self.child]:
                observationCount = observationCount + self.getTable(childValue,indices)
            # Use the count to transform the counts into conditional 
            # probabilities                
            if observationCount == 0:
                self.setProb(childValue, indices, 0)
                #print(type(self.getTable(childValue,indices)))
            else:
                for childValue in self.data.domains[self.child]:
                    self.setProb(childValue, indices, self.getTable(childValue,indices) / observationCount)
            # Increment the indices    
            indices = self.__incrementParentSetIndices(indices)
            
            
            
    def getProb(self, childValue, parentValues):
        partProb = self.prob[childValue]
        for value in parentValues:
            if len(partProb) <= value:
                print(self)
                print("getProb", len(partProb), value, self.prob[childValue], childValue, parentValues)
            partProb = partProb[value]
        return partProb
    
    def getTable(self, childValue, parentValues):
        partTable = self.table[childValue]
        for value in parentValues:
            partTable = partTable[value]
        return partTable    
    
    def setProb(self, childValue, parentValues, p):
        if parentValues == []:
            self.prob[childValue] = p
            return
        partProb = self.prob[childValue]
        i = 0
        for value in parentValues:
            i = i+1
            if i < len(self.parents):                 
                partProb = partProb[value]
            else:
                partProb[value] = p
            
    def printCPD(self):            
        print("Conditional Probability Distribution for " + str(self.child) + " with parent set " + str(self.parents) + ":")
        indices = self.__createParentSetIndices()
        while (indices != []):
            for childValue in self.data.domains[self.child]:
                line = "    P(X_"+str(self.child)+"="+str(childValue)+" | "
                i = 0
                for value in indices:
                    line = line + "X_"+str(self.parents[i])+"="+str(value)
                    i = i + 1
                    if i < len(indices):
                        line = line + ", "
                line = line + ") = " + str(self.getProb(childValue,indices))
                print(line)
            indices = self.__incrementParentSetIndices(indices)
            
    def __str__(self):
        result = "Observation counts for " + str(self.child) + " with parent set " + str(self.parents) + ":"
        if self.parents == []:
            for childValue in self.data.domains[self.child]:
                line = "\n      #Observations with X_"+str(self.child)+"="+str(childValue)                
                line = line + ": " + str(self.getTable(childValue,[]))
                result = result + line            
            result = result + "\n    Conditional Probability Distribution for " + str(self.child) + " with parent set " + str(self.parents) + ":"
            for childValue in self.data.domains[self.child]:
                line = "\n      P(X_"+str(self.child)+"="+str(childValue)
                line = line + ") = " + str(self.getProb(childValue,[]))
                result = result + line
            return result
        indices = self.__createParentSetIndices()
        while (indices != []):
            for childValue in self.data.domains[self.child]:
                line = "\n      #Observations with X_"+str(self.child)+"="+str(childValue)+", "
                i = 0
                for value in indices:
                    line = line + "X_"+str(self.parents[i])+" = "+str(value)
                    i = i + 1
                    if i < len(indices):
                        line = line + ", "
                line = line + ": " + str(self.getTable(childValue,indices))
                result = result + line
            indices = self.__incrementParentSetIndices(indices)        
        result = result + "\n    Conditional Probability Distribution for " + str(self.child) + " with parent set " + str(self.parents) + ":"
        indices = self.__createParentSetIndices()
        while (indices != []):
            for childValue in self.data.domains[self.child]:
                line = "\n      P(X_"+str(self.child)+"="+str(childValue)+" | "
                i = 0
                for value in indices:
                    line = line + "X_"+str(self.parents[i])+"="+str(value)
                    i = i + 1
                    if i < len(indices):
                        line = line + ", "
                line = line + ") = " + str(self.getProb(childValue,indices))
                result = result + line
            indices = self.__incrementParentSetIndices(indices)
        return result            


def powerset(iterable):
    print(iterable)
    #"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = iterable #list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))              
        

class GradientDescent:
    
    # Threshold: Value for the Stopping Criterion
    def __init__(self, objective, gradient, maximumIterations=10000, lineSearchIterations=20, 
                 threshold=0.00001):
        self.obj = objective
        self.grad = gradient
        self.maxLineSearch = lineSearchIterations
        self.maxIterations = maximumIterations 
        self.threshold = threshold
    
    def __backtrackingLineSearch(self, point, gradEArray):
        # Backtracking Factor
        beta = 0.5
        # Step Size Multiplier: TODO start with max step size here
        direction = self.grad.evaluate(point)
        directionA = np.array(direction)
        #print(directionA)
        directionA = -directionA
        #t = 2**20
        #for i in range(len(point)):
        #    if direction[i] > 0 and point[i]/direction[i] < t:
        #        t = point[i]/direction[i]
        #    if direction[i] < 0 and (point[i]-1) / direction[i] < t:
        #        t = (point[i]-1) / direction[i]
        stepSize = 1        
        bestStepSize = 0
        
        for i in range(self.maxLineSearch):
            if self.obj.evaluate(point + stepSize*directionA) < self.obj.evaluate(point + bestStepSize*directionA):
                if Vector(point - stepSize * gradEArray).valid(1,0) == 0:
                    bestStepSize = stepSize
            stepSize = beta*stepSize                
        return bestStepSize
    
    
    def descent(self, startVector):
        # Start of our search
        nextVector = startVector
        #
        for i in range(self.maxIterations):
            
            #print(nextVector) 
            currentVector = nextVector 
            #print("Current Vector:", currentVector)
            gradE = self.grad.evaluate(currentVector)
            gradEArray = np.array(gradE)
            stepSize = self.__backtrackingLineSearch(currentVector, gradEArray)
            #print("Step Size:", stepSize)


            #if Vector(currentVector - stepSize * gradEArray).valid(1,0) == -1 :
            #    break
            nextVector = currentVector - stepSize * gradEArray
            #print("Next Vector:", nextVector)
            changeVector = nextVector - currentVector            
            # 2-Norm of the change vector
            if len(changeVector) > 1:
                changeAmount = np.linalg.norm(changeVector,ord=2)
            else:
                changeAmount = abs(changeVector)
            #print("Change Amount:",changeAmount)    
            if changeAmount <= self.threshold:
                break
            
           
        return nextVector
        
class Constant:  
    
    def __init__(self, value):
        self.v = value
        
    def __str__(self):
        return str(self.v)
        
    def evaluate(self, x):
        return self.v  
    
    def derive(self, varindex):
        return Constant(0)
    
    def simplify(self):
        dummy = 0
    
class Variable:
    
    def __init__(self, index):
        self.i = index
        
    def __str__(self):
        return "x_" + str(self.i)
        
    def evaluate(self, x):
        return x[self.i]      
    
    def derive(self, varindex):
        if self.i == varindex:
            return Constant(1)
        return Constant(0)
    
    def simplify(self):
        dummy = 0    
    
class Power:

    def __init__(self, base, exponent):
        self.b = base
        self.e = exponent
        
    def __str__(self):
        return str(self.b) + "^" + str(self.e)
        
    def evaluate(self, x):
        return self.b.evaluate(x)**self.e.evaluate(x)
    
    def derive(self, varindex):
        return Product([self.e, Power(self.b, self.e - 1), self.b.derive(varindex)])
    
    def simplify(self):
        self.b.simplify()
        self.e.simplify()

class Sum:    
    
    def __init__(self, summands):
        self.s = summands
        
    def __str__(self):
        result = ""
        first = 1
        for summand in self.s:
            if first:
                result = str(summand)
                first = 0
            else:                       
                result = result + " + " + str(summand)
        return result
        
    def evaluate(self, x):
        sum = 0
        for summand in self.s:
            sum = sum + summand.evaluate(x)
        return sum
    
    def derive(self, varindex):
        terms = []
        for summand in self.s:
            terms.append(summand.derive(varindex))
        return Sum(terms)
    
    def simplify(self):
        #print("Sum Start: ", str(self))
        if len(self.s) > 1:
            sum = 0
            nonConst = []
            for summand in self.s:
                summand.simplify()
                #print("Sum Middle A: ", summand, type(summand))    
                if type(summand) == Constant:
                    sum = sum + summand.v
                else:
                    if (type(summand) == Product and len(summand.f) == 1 and type(summand.f[0]) == Constant and summand.f[0].v == 0):
                        dummy = 0
                    else:
                        nonConst.append(summand) 
            #print("Sum Middle: ", sum, nonConst)        
            if sum == 0 and len(nonConst) > 0:
                self.s = nonConst
            else:
                if len(nonConst) == 0:
                    self.s = [Constant(sum)]
                else:
                    self.s = nonConst
                    self.s.append(Constant(sum))    
        #print("Sum End: ", str(self))            
        
class Product:    
    
    def __init__(self, factors):
        self.f = factors
        
    def __str__(self):
        if len(self.f) == 0:
            return "1"
        result = ""
        first = 1
        for summand in self.f:
            if first:
                result = str(summand)
                first = 0
            else:
                result = result + " * " + str(summand)
        return result        
        
    def evaluate(self, x):
        product = 1
        for factor in self.f:
            product = product * factor.evaluate(x)
        return product    

    def derive(self, varindex):
        if len(self.f) == 0:
            return Constant(0)        
        u = self.f[0]
        #print("U:", u)
        uprime = u.derive(varindex)        
        # print("U':", uprime)
        v = self.f[1:]
        #print("V:", ''.join(str(e) for e in v))
        if len(v) == 1:
            vprime = v[0].derive(varindex)
        else:
            vprime = Product(v).derive(varindex)
        #print("V':", vprime)    
        return Sum([Product([uprime, Product(v)]), Product([u, vprime])])
    
    def simplify(self):
        #print("Prod Start: ", str(self))
        if len(self.f) > 1:
            isZero = 0
            p = 1
            nonConst = []
            for summand in self.f:
                if type(summand) == Constant:
                    if summand.v == 0:
                        isZero = 1
                    else:
                        p = p * summand.v
                else:
                    summand.simplify()
                    nonConst.append(summand)     
            #print("Prod Middle: ", p, isZero, nonConst)        
            if isZero:
                self.f = [Constant(0)]
            else: 
                if len(nonConst) > 0:
                    self.f = nonConst
                    self.f.append(Constant(p))                
                else:
                    self.f = [Constant(p)]                
        #print("Prod End: ", str(self))
        
class Fraction:
    
    def __init__(self, nominator, denominator):
        self.n = nominator
        self.d = denominator
        
    def __str__(self):
        return "(" + str(self.n) + ") / (" + str(self.d) + ")"
        
    def evaluate(self, x):
        den = self.d.evaluate(x)
        if den == 0:
            return self.n.evaluate(x) / 0.0001
        else: 
            return self.n.evaluate(x) / den
    
    def simplify(self):        
        self.n.simplify()
        self.d.simplify()        
        
        
class Logarithm:

    def __init__(self, argument):
        self.arg = argument
        
    def __str__(self):
        return "Log(" + str(self.arg) + ")"
        
    def evaluate(self, x):
        ev = self.arg.evaluate(x)
        if ev > 0:
            return log(self.arg.evaluate(x))
        else:
            #print("Log of 0")
            return -10000
    
    def derive(self, varindex):
        argDer = self.arg.derive(varindex)
        if (type(argDer) == Constant and argDer.v == 0):
            return Constant(0)
        den = Product([self.arg, Constant(1)])
        #print("Den", den)
        return Fraction(argDer, den)
    
    def simplify(self):
        self.arg.simplify()   
    
class Vector:
    
    def __init__(self, components):
        self.c = components
        
    def __str__(self):
        result = "["
        first = 1
        for summand in self.c:
            if first:
                result = str(summand)
                first = 0
            else:
                result = result + ", " + str(summand)
        result = result + "]"        
        return result          
        
    def evaluate(self, x):
        v = []
        for comp in self.c:
            v.append(comp.evaluate(x))
        return v
    
    def derive(self, varindex):
        v = []
        for comp in self.c:
            v.append(comp.derive(varindex))
        return Vector(v)
    
    def simplify(self):
        for summand in self.c:
            summand.simplify()
    
    def valid(self, upper, lower):
        for comp in self.c:
            if comp > upper:
                return -1
            if comp < lower:
                return -1
        return 0

def testFunction(x):  
    return x[0]


def generateQOfX(child, childValue, parentSet, parentValues):
    # Equation 2
    noiseFactors = []
    for i in range(len(parentSet)):
        # if this is true, then this is in T_x
        if parentValues[i] == 1:
            noiseFactors.append(Variable(parentSet[i]))
    result = Product(noiseFactors)   
    if (childValue == 0):
        return result
    else:
        # Equation 3
        minusOne = Constant(-1)
        minusEq2 = Product([minusOne, result])
        result = Sum([Constant(1), minusEq2])
        return result
    
def getPofX(cpd, childValue, parentValues):
    return cpd.getProb(childValue, parentValues)
            
def generateKLTerm(cpd, child, childValue, parentSet, parentValues):
    pX = Constant(getPofX(cpd, childValue, parentValues))
    qX = generateQOfX(child, childValue, parentSet, parentValues)     
    pXLogpX = Product([pX, Logarithm(pX)])
    pXLogqX = Product([Constant(-1), pX, Logarithm(qX)])
    sumOfLogs = Sum([pXLogpX, pXLogqX])
    return sumOfLogs


def createParentSetIndices(parents):
    indices = []
    for d in parents:
        indices.append(0)
    return indices    

def incrementParentSetIndices(indices):
    i = len(indices)-1
    while (i >= 0 and indices[i] == 1):
        indices[i] = 0
        i = i-1
    if i == -1:
        return []
    else:
        indices[i] = indices[i] + 1
        return indices
    
    

def generateKL(cpd, child, parentSet):
    summands = []
    # We make a list that contains a counter for the instantiations of the child node 
    # and each parent node
    indices = []
    # Adds a counter for the instantiation of the child node
    indices.append(0)
    # Adds a counter for the instantiation of each parent node
    for i in range(0, len(parentSet)):
        indices.append(0)
    while (indices != []):
        childValue = indices[0]
        parentValues = indices[1:]
        summand = generateKLTerm(cpd, child, childValue, parentSet, parentValues)
        summands.append(summand)
        indices = incrementParentSetIndices(indices)
    return Sum(summands)

def generateLogLikelihoodTerm(cpd, child, childValue, parentSet, parentValues):
    #pX = Constant(getPofX(cpd, childValue, parentValues))
    # Number of observations:
    noo = cpd.getTable(childValue, parentValues) * -1.0
    if noo == 0:
        return Constant(0)
    # q(X)
    qX = generateQOfX(child, childValue, parentSet, parentValues)   
    # Combined:
    expqX = Product([Constant(noo), Logarithm(qX)])
    return expqX

def generateLogLikelihood(cpd, child, parentSet):
    summands = []
    # We make a list that contains a counter for the instantiations of the child node 
    # and each parent node
    indices = []
    # Adds a counter for the instantiation of the child node
    indices.append(0)
    # Adds a counter for the instantiation of each parent node
    for i in range(0, len(parentSet)):
        indices.append(0)
    while (indices != []):
        childValue = indices[0]
        parentValues = indices[1:]
        summand = generateLogLikelihoodTerm(cpd, child, childValue, parentSet, parentValues)
        summands.append(summand)
        indices = incrementParentSetIndices(indices)
    return Sum(summands)
    
def snoob(x): 
      
    next = 0
    if(x): 
          
        # right most set bit 
        rightOne = x & -(x) 
          
        # reset the pattern and  
        # set next higher bit 
        # left part of x will  
        # be here 
        nextHigherOneBit = x + int(rightOne) 
          
        # nextHigherOneBit is  
        # now part [D] of the  
        # above explanation. 
        # isolate the pattern 
        rightOnesPattern = x ^ int(nextHigherOneBit) 
          
        # right adjust pattern 
        rightOnesPattern = (int(rightOnesPattern) / 
                            int(rightOne)) 
          
        # correction factor 
        rightOnesPattern = int(rightOnesPattern) >> 2
          
        # rightOnesPattern is now part 
        # [A] of the above explanation. 
          
        # integrate new pattern  
        # (Add [D] and [A]) 
        next = nextHigherOneBit | rightOnesPattern 
    return next

def BIC(childNodeIndex, parentSet, noiseparameters, cpd):

    return 0
def pruneTK():

    return 0

def pruneTKextended():

    return 0

def scoreParentSet(childNodeIndex, candidateParentSet, data, startPoint):
    print(candidateParentSet)
    cpd = data.computeCPD(childNodeIndex, candidateParentSet)

    objectiveKL = generateKL(cpd, childNodeIndex, candidateParentSet)
    objectiveLL = generateLogLikelihood(cpd, childNodeIndex, candidateParentSet)

    gradientSize = max(candidateParentSet) +1
    gradientsKLList = [Constant(0)] * gradientSize
    ###gradientsLLList = [Constant(0)] * gradientSize

    #print(objectiveLL)
            
    for g in range(gradientSize):
        #if candidateParentSet.count(g) == 0:
        #    gradientsKLList[g] = Constant(0)
        #    gradientsKLList[g] = Constant(0)

        if candidateParentSet.count(g) == 1:
            gradientTempKL = objectiveKL.derive(g)
            gradientTempKL.simplify()
            ####gradientTempLL = objectiveLL.derive(g)
            ###gradientTempLL.simplify()
            gradientsKLList[g] = gradientTempKL
            ###gradientsLLList[g] = gradientTempLL

            #else:
            #    print("error : incorrect parent set formed")
    gradientsKL = Vector(gradientsKLList)
    ###gradientsLL = Vector(gradientsLLList)
    ###startPointKL = [0] * gradientSize
    ###startPointLL = [0] * gradientSize 
            
    gradientDescentKL = GradientDescent(objectiveKL, gradientsKL)
    gresultKL = gradientDescentKL.descent(startPoint)
    #print(gresultKL)
    #print("Objective: ", objectiveLL)
    #print("Gradient: ", gradientsLL)

    ####gradientDescentLL = GradientDescent(objectiveLL, gradientsLL)
    ###gresultLL = gradientDescentLL.descent(startPointLL)
    #print(gresultKL)
    
    llTermOfBic = objectiveLL.evaluate(gresultKL)
    penalty = (2**len(candidateParentSet))*log(data.numberOfRows)/2 #penalty
    print(llTermOfBic+ penalty)
    #print(llTermOfBic)
    return llTermOfBic+penalty, gresultKL
    


def checkAdditionToPoisoned(poisoned, candidateParentSet, bic, data, maxSize):
    sizeHorizon = ceil(log(2*bic/log(data.numberOfRows)))
    if (sizeHorizon < maxSize):
        print("poisoned")
        poisoned[toNumber(candidateParentSet)] = sizeHorizon

def sublist(ls1, ls2):
    '''
    >>> sublist([], [1,2,3])
    True
    >>> sublist([1,2,3,4], [2,5,3])
    True
    >>> sublist([1,2,3,4], [0,3,2])
    False
    >>> sublist([1,2,3,4], [1,2,5,6,7,8,5,76,4,3])
    False
    '''
    def get_all_in(one, another):
        for element in one:
            if element in another:
                yield element

    for x1, x2 in zip(get_all_in(ls1, ls2), get_all_in(ls2, ls1)):
        if x1 != x2:
            return False

    return True

def toNumber(numberList):
    result = 0
    for number in numberList:
        result = result | (1 << number)
    return result     
    

def notPoisoned(candidateParentSet, poisoned):    
    for key in poisoned.keys():
        if key & toNumber(candidateParentSet) == key and poisoned[key] < len(candidateParentSet):
            return False
    return True    

def scoreNodeWithEmptyParentSet(child, data):
    cpd = data.computeCPD(child, [])
    #print(cpd)
    llTermOfBic = - cpd.getTable(0, []) * log(cpd.getProb(0, [])) - cpd.getTable(1, []) * log(cpd.getProb(1, []))
    penalty = log(data.numberOfRows) / 2 #penalty

    #print(llTermOfBic, penalty)
    return llTermOfBic + penalty    
    
def scoreNode(childNodeIndex, data, maxSize):
    
    print("scoring node :", childNodeIndex)
    nullscore = scoreNodeWithEmptyParentSet(childNodeIndex, data)
    print("null parent score = ", nullscore)
    totalVar = data.numberOfColumns
    
    parentPowerSet = list(range(totalVar))
    parentPowerSet.remove(childNodeIndex)
    
    scoreMap = {}
    poisoned = {}

    for i in range(1,min(len(parentPowerSet),maxSize)):
        parentPowerSetsi = list(it.combinations(parentPowerSet,i))
        
        for p in parentPowerSetsi:
            candidateParentSet = list(p)
            if notPoisoned(candidateParentSet, poisoned):
                
                bic = scoreParentSet(childNodeIndex, candidateParentSet, data)
                print(candidateParentSet, bic)
                if (bic < 10000):
                    print("parent set:", candidateParentSet, bic)
                scoreMap[toNumber(candidateParentSet)] = bic
                checkAdditionToPoisoned(poisoned, candidateParentSet, bic, data, maxSize)
            
    return 0  



class PruneTreeNode:
    
    def __init__(self, hashMap, nextQueue, parentTreeNode, data, childNodeIndex, numberOfVariables, parentSet, score, X, 
                 completePruning, outputfile, outputfile1, noiseParams):
        self.noiseParams = noiseParams
        #print("Scored: ", parentSet, score)
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
        return 2**sizeOfSet * log(self.data.numberOfRows) / 2.0 #penalty
    
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
                    print("Ignoring unary node:", i)
                
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
                    if self.hashMap[subsetNumber] < childPenalty - self.X:
                        #print("  Pruned all children of", self.parentSet)
                        return
                    if self.hashMap[subsetNumber] < bestSubsetScore:
                        bestSubsetScore = self.hashMap[subsetNumber]
                        bestSubset = subsetNumber
        ###print("    Creating child", childParentSet, bestSubsetScore, bestSubset)
        self.createChild(childParentSet, bestSubsetScore, bestSubset, outputfile, outputfile1)                
        
                
    def createChild(self, childParentSet, bestScoreAbove, bestSubsetAbove, outputfile, outputfile1):
        ###print("Create child", i)
        fo = open(outputfile,"a+")
        fo1 = open(outputfile1,"a+")

        maxC = max(childParentSet)
        startPoint = list(self.noiseParams)
        while (len(startPoint) < maxC+1):
            startPoint.append(0)        
        
        bic, noiseParams = scoreParentSet(self.childNodeIndex, childParentSet, self.data, startPoint)
        #print(bic)
        #print(bestScoreAbove)
        if (bic < bestScoreAbove + self.X):
            print("Adding:", childParentSet, bic)
            self.hashMap[self.__createNumber(childParentSet)] = bic
            output = "   " + str(-bic) + " " + str(len(childParentSet))
            output1 = str(len(childParentSet))

            for n in childParentSet:
                output = output + " " + str(n)
                output1 = output1 + " " + str(n)

            for n in childParentSet:
                output1 = output1 + " " + str(noiseParams[n])

            fo.write(output + "\n")
            fo1.write(output1 + "\n") 
        child = PruneTreeNode(self.hashMap, self.nextQueue, self, self.data, self.childNodeIndex, self.numberOfVariables, 
                              childParentSet, bic, self.X, self.completePruning, self.outputfile, self.outputfile1, noiseParams)
        self.children.append(child) 
        self.nextQueue.append(child)        
        ###print("Created", i)
        fo.close()
        fo1.close()
        

def score(data , nodeID, maxSize, X, outputfile, outputfile1):
    totalVar = data.numberOfColumns
    #print("Parent sets for child node", nodeID)
    hm = {}
    queue = deque()
    nextQueue = deque()        
    #print(" Creating parent sets with 0 nodes:") 
    s = scoreNodeWithEmptyParentSet(nodeID,data)
    hm[0] = s        
    ###print(hm)

    fo = open(outputfile,"a+")
    output = "   " + str(-s) + " " + str(0) +"\n"
    fo.write(output)
    fo.close()

    fo1 = open(outputfile1,"a+")
    output = str(0) +" 0\n"
    fo1.write(output)
    fo1.close()

    print("  Parent set", [], s)

    root = PruneTreeNode(hm, nextQueue, None, data, nodeID, totalVar, [], s, X, True, outputfile,outputfile1, [])  
    queue.append(root)
    level = 2
    #print(" Creating parent sets with 1 node:") 
    while (queue or nextQueue):            
        if not queue:
           # print(" Creating parent sets with", level, "nodes:") 
            level = level + 1
            for i in nextQueue:
                queue.appendleft(i)
            nextQueue.clear()
        ###print("Pop")    
        currentNode = queue.popleft() 
        
        #print(data.domains[currentNode])
        currentNode.createChildren(outputfile, outputfile1)
    print("Done.")
            
        #scoreNode(i, data, maxSize)            





d = Data();
d.readFromCSVFile(sys.argv[1])
childParentSet = list(range(1,d.numberOfColumns))

outfile = sys.argv[3] + "_" + sys.argv[2] + "_" + sys.argv[4]
outfile1 = sys.argv[3] + "_" + sys.argv[2] + "_" + sys.argv[4] + "_noise"
print(outfile)
fo = open(outfile,"w+")
fo1 = open(outfile1,"w+")
fo.close() 
fo1.close()  
bf = log(int(sys.argv[4]))
#bic, noise1 = scoreParentSet(4,[3,13,20],d, [0]*21)
#print(bic)
#print(noise1)
#bic, noise1 = scoreParentSet(4,[3,13,20],d, [0.1]*21)
#print(bic)
#print(noise1)
#bic, noise1 = scoreParentSet(4,[3,13,20],d, [0.3]*21)
#print(bic)
#print(noise1)
#bic, noise1 = scoreParentSet(4,[3,13,20],d, [0.5]*21)
#print(bic)
#print(noise1)
#bic, noise1 = scoreParentSet(4,[3,13,20],d, [0.7]*21)
#print(bic)
#print(noise1)

#sys.exit(0)
score(d, int(sys.argv[2]), d.numberOfRows, bf, outfile, outfile1)


#bic, noise1 = scoreParentSet(0,list(range(1,d.numberOfColumns)),d, [0.1]*d.numberOfColumns)
#bic, noise2 = scoreParentSet(0,list(range(1,d.numberOfColumns)),d, [0.3]*d.numberOfColumns)
#bic, noise3 = scoreParentSet(0,list(range(1,d.numberOfColumns)),d, [0.5]*d.numberOfColumns)
#bic, noise4 = scoreParentSet(0,list(range(1,d.numberOfColumns)),d, [0.7]*d.numberOfColumns)
#bic, noise5 = scoreParentSet(0,list(range(1,d.numberOfColumns)),d, [0.9]*d.numberOfColumns)

#ferr = open(sys.argv[3]+"_noise", "r")
#lines=ferr.readlines()
#for l in lines:
#    if l[0] =="0":
#        np = [float(i) for i in l.split()]
#        nps=np[2::2]
#err1 = 0
#err2 = 0
#err3 = 0
#err4 = 0
#err5 = 0

#for i in range(0,len(nps)):
#    err1 += abs(nps[i] - noise1[i+1])/nps[i]
#    err2 += abs(nps[i] - noise2[i+1])/nps[i]
#    err3 += abs(nps[i] - noise3[i+1])/nps[i]
#    err4 += abs(nps[i] - noise4[i+1])/nps[i]
#    err5 += abs(nps[i] - noise5[i+1])/nps[i]
#errs = [err1,err2,err3,err4,err5]
#noises = [noise1, noise2, noise3, noise4, noise5]
#err=min(errs)
#print(err)
#best = errs.index(err)
#bestnoise=noises[best]
#outfile1 = sys.argv[3] + "_" + sys.argv[2] + "_" + sys.argv[4] + "_noise"
#fo1 = open(outfile1,"w+")
#output1 = str(len(childParentSet))

#for c in childParentSet:
#    output1 = output1 + " " + str(c)

#for c in childParentSet:
#    output1 = output1 + " " + str(bestnoise[c])

#fo1.write(output1 + "\n") 

#fo1.close() 
#f=open("ERR_GD_"+ str(len(childParentSet)+1) + "_" + str(d.numberOfRows), "a+")
#f.write(str(err/len(nps)))
#f.write("\n")
#f.close()

#sys.exit(0)
#score(d, int(sys.argv[2]), d.numberOfRows, bf, outfile, outfile1)
#scoreNode(int(sys.argv[2]), d, d.numberOfRows)
#print("Done scoring")

#scoreParentSet(0, [16], d)
#score(d,5)
#childNodeIndex = 2
#parentSet = [0, 1]
#childNodeIndex = 3
#parentSet = [0, 1,2]
#startPoint = [0.1,0.1,0.1,0.1,0.1]
#

#childNodeIndex1 = 0
#parentSet1 = [1,2,3]

#print("")
#print("")
#print("File Mushroom.csv has been loaded successfully.")
#print("  Number of variables:", d.numberOfColumns)
#print("  Number of observations:", d.numberOfRows)
#print("")
#print("  Child node:", childNodeIndex)
#print("  Parent set:", parentSet)
#print("")
#print("  Computing CPD:")
#cpd = d.computeCPD(childNodeIndex,parentSet)
#print("    CPD:", cpd)

#cpd.setProb(0 , [0,0], 1)
#cpd.setProb(0 , [0,1], 0.5)
#cpd.setProb(0 , [1,0], 0.5)
#cpd.setProb(0 , [1,1], 0.25)

#cpd.setProb(1 , [0,0], 0)
#cpd.setProb(1 , [0,1], 0.5)
#cpd.setProb(1 , [1,0], 0.5)
#cpd.setProb(1 , [1,1], 0.75)

#cpd1 = d.computeCPD(childNodeIndex1,parentSet1)
##cpd1.setProb(0 , [0,0,0], 1)
#cpd1.setProb(0 , [0,0,1], 0.1)
#cpd1.setProb(0 , [0,1,0], 0.2)
#cpd1.setProb(0 , [0,1,1], 0.02)
#cpd1.setProb(0 , [1,0,0], 0.6)
#cpd1.setProb(0 , [1,0,1], 0.06)
#cpd1.setProb(0 , [1,1,0], 0.12)
#cpd1.setProb(0 , [1,1,1], 0.012)

#cpd1.setProb(1 , [0,0,0], 0)
#cpd1.setProb(1 , [0,0,1], 0.9)
#cpd1.setProb(1 , [0,1,0], 0.8)
#cpd1.setProb(1 , [0,1,1], 0.98)
#cpd1.setProb(1 , [1,0,0], 0.4)
#cpd1.setProb(1 , [1,0,1], 0.94)
#cpd1.setProb(1 , [1,1,0], 0.88)
#cpd1.setProb(1 , [1,1,1], 0.988)


#print("    CPD:", cpd)

#print("")
#print("Q(0,0,0) = ", generateQOfX(0,0,[1,2],[0,0]))
#print("Q(1,1,1) = ", generateQOfX(0,1,[1,2],[1,1]))
#print("Q(0,1,0) = ", generateQOfX(0,0,[1,2],[1,0]))
#print("Q(0,1,1) = ", generateQOfX(0,0,[1,2],[1,1]))

#print("KL(0,0,0) = ", generateKLTerm(cpd,0,0,[1,2],[0,0]))
#print("KL(0,0,1) = ", generateKLTerm(cpd,0,0,[1,2],[0,1]))
#print("KL(0,1,1) = ", generateKLTerm(cpd,0,0,[1,2],[1,1]))
#print("KL(1,1,1) = ", generateKLTerm(cpd,0,1,[1,2],[1,1]))



#obj = generateKL(cpd,childNodeIndex,parentSet)
#obj1 = generateLogLikelihood(cpd,childNodeIndex,parentSet)

#obj = generateLogLikelihood(cpd,10,[0,1,2,3,4,5,6,7,8,9])
#obj = generateKL(cpd,2,[0,1])
#print("Obj:", obj)
#gradient1 = obj.derive(0)
#gradient2 = obj.derive(1)
#gradient3 = obj.derive(2)
#gradient4 = obj.derive(3)
#gradient5 = obj.derive(4)

#print(obj1.evaluate([0,0,0]))


   



