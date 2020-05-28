from math import sqrt
from DataClass import Data
from random import uniform, sample
import sys
class EM:
    
    def __init__(self, data, childIndex, parentIndices, threshold = 0.001, maxIterations = 100):        
        self.data = data    
        self.childIndex = childIndex
        self.parentIndices = parentIndices
        self.threshold = threshold
        self.maxIterations = maxIterations
        
    def numberOfObservations(self):
        return len(self.data.data)
    
    def countObservations(self, parentIndex):
        count = 0
        for row in self.data.data:
            if row[parentIndex] == 1:
                count = count + 1
        return count
    
    def observations(self):
        return self.data.data
        
    def __computeParentCPNew(self, parentIndex, row, currentParameters):
        y = row[self.childIndex]
        xi = row[parentIndex]
        pi = currentParameters[parentIndex]
        
        product = 1
        for parentI in self.parentIndices:
            #print(parentI, self.parentIndices, row[parentI])
            if row[parentI] == 1:
                #print(currentParameters[parentI])
                product = product * currentParameters[parentI]
        denominator = 1 - product    
        
        if y == 1 and xi == 1:
            return False, 0
        if y == 0 and xi == 0:
            return False, 0
        if y == 1 and xi == 0:
            return False, 1 - denominator
        if y == 0 and xi == 1:
            return False, 1
        
        
        nominator = y * xi * pi
        
        
                
        
        if denominator == 0:
            #print(product)
            return True, 0
        else:
            return False, nominator / denominator         
    
    def __computeParentCP(self, parentIndex, row, currentParameters):
        y = row[self.childIndex]
        xi = row[parentIndex]
        pi = currentParameters[parentIndex]
        nominator = y * xi * pi
        product = 1
        for parentI in self.parentIndices:
            #print(parentI, self.parentIndices, row[parentI])
            if row[parentI] == 1:
                #print(currentParameters[parentI])
                product = product * (1 - currentParameters[parentI])
        denominator = 1 - product    
        if denominator == 0:
            #print(product)
            return True, 0
        else:
            return False, nominator / denominator        

    def __computeParentParameter(self, parentIndex, currentParameters):
        result = 0
        for row in self.observations():
            #print("Row: ", row)
            bad, value = self.__computeParentCPNew(parentIndex, row, currentParameters)
            result = result + value
            #if bad:
            #    return True, 0
            #else:
            #    result = result + value
        return False, result / self.countObservations(parentIndex)    
        #'return False, result / self.numberOfObservations()
    
    def __computeParentParameters(self, currentParameters):
        updatedParameters = [0] * len(currentParameters)
        for parentIndex in self.parentIndices:
            bad, updatedParameters[parentIndex] = self.__computeParentParameter(parentIndex, currentParameters)
            #if bad:
            #    return True, [0] * len(currentParameters)
        return False, updatedParameters
    
    def __distance(self, list1, list2):
        sum = 0
        for i in range(len(list1)):
            sum = sum + (list1[i] - list2[i]) ** 2
        return sqrt(sum)
    
    def compute(self, startParameters):        
        #print("Start: ", startParameters)
        bad, currentParameters = self.__computeParentParameters(startParameters)
        iterations = 0
        #print("Initial: ", bad, currentParameters)
        while (self.__distance(startParameters, currentParameters) > self.threshold and iterations < self.maxIterations):
            startParameters = currentParameters
            bad, currentParameters = self.__computeParentParameters(startParameters)
            iterations = iterations + 1
            #print("Loop: ", currentParameters)
        return bad, currentParameters    
    
#data = Data()
#data.readFromCSVFile(sys.argv[1])
#data.readFromCSVFile("f_3_10000_2.csv") # 0.1, 0.05 
#data.readFromCSVFile("f_03_08_3_100.csv")

#em = EM(data, 0, [1,2])
#x, y = em.compute([0.01, 0.3, 0.8]) 
#print("Result", y)
#for i in range(1):
#    startPoint = [uniform(0,1) for _ in range(3)]
#    x, y = em.compute(startPoint) 
#   print("Start Point", startPoint, "Result", y)
