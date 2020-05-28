import os
import numpy as np
from numpy import genfromtxt
import sys
import networkx as nx
from random import randint, uniform
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
    
    def findBestOrderForChildNode(self, childNode):
        currentMin = 10000
        bestRowIndex = -1
        rowIndex = -1
        for row in self.data:
            rowIndex = rowIndex + 1 
            if row[childNode] == 1:
                rowCount = 0
                for i in range(len(row)):
                    if row[i] > 0:
                        rowCount = rowCount + 1
                if rowCount < currentMin:
                    bestRowIndex = rowIndex
                    currentMin = rowCount
                    
        result = []                    
        for i in range(len(self.data[bestRowIndex])):
            if i != childNode and self.data[bestRowIndex][i] > 0:
                result.append(i)
        mandatory = len(result)        
        for i in range(len(self.data[bestRowIndex])):
            if i != childNode and self.data[bestRowIndex][i] == 0:
                result.append(i)
        return mandatory, result        
        

class CPD:
    
    def __init__(self, data, childNode, parentSet, filename):
        self.data = data
        self.child = childNode
        self.parents = parentSet
        self.filename= filename
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
            else:
                for childValue in self.data.domains[self.child]:
                    self.setProb(childValue, indices, self.getTable(childValue,indices) / observationCount)
            # Increment the indices    
            indices = self.__incrementParentSetIndices(indices)
            
    def getProb(self, childValue, parentValues):
        partProb = self.prob[childValue]
        for value in parentValues:
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
            
    def __indicesToStr(self, indices):
        line = "("
        first = True
        for i in indices:
            if first:
                first = False
            else:
                line = line + ", "
            if i == 1:
                line = line + "yes"
            else:
                line = line + "no"                
        return line + ")"            
            
    def printBiff(self, filename):
        
            if len(self.data.domains[self.child]) == 1:
                f = open(filename, "a")
                f.flush()
                if self.data.domains[self.child]==1:
                    f.write("  table 1.0 0.0;\n")
                else :
                    f.write("  table 0.0 1.0;\n")
                f.close()
                return
            #try
            if self.parents == []:
                f = open(filename, "a")
                f.flush()
                f.write("  table " + str(self.getProb(0, [])) + ", "+ str(self.getProb(1, [])) + ";\n")
                f.close()
                return
            indices = self.__createParentSetIndices()
            f = open(filename, "a")
            f.flush()
            while (indices != []):
                line = "  " + self.__indicesToStr(indices) + " " + str(self.getProb(0, indices)) + ", " + str(self.getProb(1, indices)) + ";\n" 
                f.write(line)
                indices = self.__incrementParentSetIndices(indices)
            f.flush()
            f.close() 
        #except IOError as (errno,strerror):
            #print "I/O error({0}): {1}".format(errno, strerror) 
                        
            
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

class NoisyORCPD:
    
    # Create with empty parent set
    def __init__(self, child):
        self.child = child
        self.parents = []
        self.dimensions = [2]
        self.prob = np.ndarray(shape=self.dimensions)
        self.prob.fill(0)
        self.setProb(0,[], 0.5)
        self.setProb(1,[], 0.5)
        self.outfile = ""
        
    # Create with empty parent set
    def __init__(self, noiseParameters, child, parents):
        self.child = child
        self.parents = parents
        self.dimensions = [2]
        for parent in parents:
            self.dimensions.append(2)            
        self.prob = np.ndarray(shape=self.dimensions)
        self.prob.fill(0)
        
        if self.parents == []:
            self.setProb(0, [], 0.5)
            self.setProb(1, [], 0.5)
            return
        indices = self.__createParentSetIndices()
        while (indices != []):
            product = 1
            for i in range(len(indices)):
                if indices[i] == 1:
                    product = product * noiseParameters[parents[i]]
            probIsOne = 1 - product
            probIsZero = product
            
            self.setProb(1, indices, probIsOne)
            self.setProb(0, indices, probIsZero) 
            
            indices = self.__incrementParentSetIndices(indices)

    
    def __createParentSetIndices(self):
        indices = []
        for d in self.parents:
            indices.append(0)
        return indices    

    def __incrementParentSetIndices(self, indices):
       i = len(indices)-1
       while (i >= 0 and indices[i] == 1):
           indices[i] = 0
           i = i-1
       if i == -1:
           return []
       else:
           indices[i] = indices[i] + 1
           return indices

            
    def getProb(self, childValue, parentValues):
        partProb = self.prob[childValue]
        for value in parentValues:
            partProb = partProb[value]
        return partProb
    
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
                
    def __indicesToStr(self, indices):
        line = "("
        first = True
        for i in indices:
            if first:
                first = False
            else:
                line = line + ", "
            if i == 1:
                line = line + "yes"
            else:
                line = line + "no"                
        return line + ")"
            
    def printCPD(self, outfile):  
        fo = open(outfile, "a")
        if self.parents == []:
            fo.write("  table 0.5, 0.5; \n")
            return
        #print("Alloha")
        indices = self.__createParentSetIndices()
        #print("Alohb")
        while (indices != []):
            #print(indices)
            line = "  " + self.__indicesToStr(indices) + " " + str(self.getProb(0, indices)) + ", " + str(self.getProb(1, indices)) + ";"                
            fo.write(line + "\n")
            indices = self.__incrementParentSetIndices(indices)
        fo.close()
            
    def __str__(self):
        result = "Conditional Probability Distribution for " + str(self.child) + " with parent set " + str(self.parents) + ":"
        if self.parents == []:        
            for childValue in [0,1]:
                line = "\n      P(X_"+str(self.child)+"="+str(childValue)
                line = line + ") = " + str(self.getProb(childValue,[]))
                result = result + line
            return result     
        result = "Conditional Probability Distribution for " + str(self.child) + " with parent set " + str(self.parents) + ":"
        indices = self.__createParentSetIndices()
        while (indices != []):
            for childValue in [0,1]:
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

def random_dag(nodes, averageDegree, maximumInDegree):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    edges = nodes * averageDegree / 2
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = randint(0,nodes-1)
        b = a
        while b==a or G.in_degree(b) == maximumInDegree:
            b = randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G

def printStats(G):
    maxInDegree = 0
    degreeSum = 0
    for node in G.nodes:
        degreeSum = degreeSum + G.degree(node)
        if G.in_degree(node) > maxInDegree:
            maxInDegree = G.in_degree(node)
    print("Number of nodes:", G.number_of_nodes())            
    print("Number of edges:", G.number_of_edges())            
    print("Maximum in-degree:", maxInDegree)
    print("Average degree:", degreeSum * 1.0 / G.number_of_edges())

def flipWithNoise(p, val):
    fl =  np.random.binomial(1, p)
    if fl==1:
        if val==1:
            return 0
        else : 
            return 1
    return val

def readNoisyORParameters(dataset, noisefile, child, pred, n,  outfile,):
    freg = open(noisefile, "r")
    f=""
    if os.stat(noisefile).st_size > 0 :
        f = freg.readlines()
    

    parentfound = False
    print(noisefile)
    
    for line in f :
        noiseParameters = [0] * n
        items = [float(it) for it in line.split()]
        splitindex = int(items[0])+1
        if len(items) > 2:
            #print(items[1:splitindex])
            parents = [int(p) for p in items[1:splitindex]]
            noise = items[splitindex:]
            #print(parents)
            #print(noise)
            for i in range(0,int(items[0])):
                noiseParameters[parents[i]] = noise[i]
                
            #check if this edge exists in graph
            if parents==pred :
                cpd = NoisyORCPD(noiseParameters, child, pred)
                cpd.printCPD(outfile)
                parentfound = True
                print(child)
                print("regular")
 
    freg.close()
    if parentfound == False: 
        data =  Data();
        data.readFromCSVFile(dataset + ".csv")
        cpd = CPD(data, child, pred,outfile)    
        cpd.printBiff(outfile)
        print(child)
        print("noisy")


    
def bicToBiff(bicfile,dataset, n, netN):
    newfile = True
    filectr=0
    f = [None]*netN
    for i in range(0,netN):
        f[i] = open(dataset + "_net_" + str(i), "w+")
        f[i].write("network unknown {\n")        
        f[i].write("}\n")
        for j in range(0, n):
            f[i].write("variable v"+str(j)+" {\n")
            f[i].write("    type discrete [ 2 ] { no, yes };\n")
            f[i].write("}\n")    
        f[i] .close()
    freg = open(bicfile, "r")
    lines = freg.readlines()
    
    
    prefixes = ('\n')
    suffixes = ('BN')
    
    for line in lines:
        if line.startswith(prefixes):
            continue
        if line.startswith(suffixes):
            
            newfile = True
            filectr+=1
            continue
        if len(line) < 6:
            continue
  
        #read Graph
        parts = line.split()
        subparts = parts[0].split("<-")
        child = subparts[0]
        if len(subparts) <= 1:
            parentsT = []
        else:
            parentsT = subparts[1].split(",")
        #print(child)    
        #print(parentsT)            
        parents = []
        child = int(child)
        for p in parentsT:
            if p != '':
                parents.append(int(p))
           
        outfile = dataset + "_net_" + str(filectr)
        fl  = open(outfile, "a")
        
        if parents == []:
           tr= fl.write("probability ( v"+str(child)+" ) {\n")  
           #print(tr)
        else:
            string = "v"+str(child)+" | "
            first = True
            for p in parents:
                if first:
                    first = False
                else:
                    string = string + ", "
                string = string + "v"+str(p)
            fl.write("probability ( "+string+" ) {\n")
            #print(tr)
        
        fl.close()
        print(child)
        print(parents)
        noisefile = dataset + "_" + str(child) + "_20_noise"        
        readNoisyORParameters(dataset, noisefile, child , parents,n, outfile)
        fl = open(outfile, "a")
        

        fl.write("}\n")
        fl.close()

gobnilpfile = sys.argv[1] + ".merged.gobnilp"
bicToBiff(gobnilpfile, sys.argv[1],int(sys.argv[3]), int(sys.argv[4]))
    





