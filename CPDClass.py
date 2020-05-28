import numpy as np

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
