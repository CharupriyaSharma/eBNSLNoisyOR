from BIFClass import BIFFile
from DataClass import Data
from PruneTreeNodeClass import createIndices, incrementIndices
from math import log
import sys

class ConditionalKLDivergence:
    
    def __init__(self, bif1, bif2, data):
        self.bif1 = bif1
        self.bif2 = bif2
        self.data = data
        self.child1 = self.__getVariableWithParents(bif1)        
        self.child2 = self.__getVariableWithParents(bif2)  
        parents = list(range(1, data.numberOfColumns))
        self.cpd = self.data.computeCPD(0, parents)
    
    def __getVariableWithParents(self, bif):
        for v1 in bif.variables.values():
            if not v1.parents == []:
                return v1
            
    def __computeKLTerm(self, childValue, parentValues):
        pb = self.cpd.getProb(0, parentValues) + self.cpd.getProb(1, parentValues)
        p1ab = float(self.child1.getProb(childValue, parentValues))
        p2ab = float(self.child2.getProb(childValue, parentValues))
        return pb * p1ab * log(p1ab / p2ab)
    
    def compute(self):                
        indices = createIndices(self.child1.parents)
        s = 0
        while (indices != []):
            childValue = indices[0]
            parentValues = indices[1:]
            s = s + self.__computeKLTerm(childValue, parentValues)
            indices = incrementIndices(indices)
        return s
    
bif1 = BIFFile(sys.argv[1])
bif2 = BIFFile(sys.argv[2])
data = Data()
data.readFromCSVFile(sys.argv[3]) 
ckld = ConditionalKLDivergence(bif1, bif2, data)
f=open(sys.argv[4], "a+")
f.write(str(ckld.compute()))
f.write("\n")
f.close()
