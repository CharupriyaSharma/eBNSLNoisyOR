import numpy as np
import sys
import networkx as nx
from math import floor
from random import uniform

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
def fixed_dag(nodes):
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    for i in range(1,nodes):
        G.add_edge(i,0)
    return G 


def random_dag(nodes, averageDegree, maximumInDegree):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    edges = nodes * averageDegree / 2
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = np.random.randint(0,nodes)
        b = a
        while b==a or G.in_degree(b) == maximumInDegree:
            b = np.random.randint(0,nodes)
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
# Parameters for the Noisy OR Network Generation and 
numberOfVariables = 10
averageSizeOfParentSet = 4    

sampleSize = 100
debug = False

numberOfVariables = int(sys.argv[1])
sampleSize = int(sys.argv[2])
sampleNoise = 0.01
#G = random_dag(numberOfVariables, 2, 5)
G=fixed_dag(numberOfVariables)
outfile = sys.argv[3]+ "_"+  sys.argv[1] + "_" + str(sys.argv[2]) 
fcsv = open(outfile + ".csv", "w+")
fcsv2 = open(outfile + ".vcsv", "w+")
fevidence = open( outfile + ".evidence.csv", "w+")
fbif = open(outfile + ".bif", "w+")
fkl = open(outfile + ".kl", "w+")

for i in range(1,numberOfVariables):
    fcsv2.write("X"+str(i)+",")
fcsv2.write("Y\n")


# Convert to node and edge numbers
# Generate the network

# Topological Sort of the network
sortedG = nx.topological_sort(G)
variablesWithNoisyParentSets = list(range(0, numberOfVariables))
for node in nx.topological_sort(G):
    pred = [n for n in G.predecessors(node)]
    if len(pred) == 0:
        variablesWithNoisyParentSets.remove(node)

noEmpty = numberOfVariables - len(variablesWithNoisyParentSets)
noSuppressed = floor(max(0, 0.7 * numberOfVariables - noEmpty))
for i in range(0, noSuppressed):
    j = np.random.randint(0, len(variablesWithNoisyParentSets))
    variablesWithNoisyParentSets.pop(j)

noisyParameters = []
for i in range(0, numberOfVariables):
    value = uniform(0.01,1)
    
    value = round(value, 2)
    noisyParameters.append(value)
#noisyParameters = [0.0, 0.2, 0.1, 0.3]

npfile = open(outfile + "_noise", "w+")
for node in nx.topological_sort(G):
    npfile.write(str(node))
    if node in variablesWithNoisyParentSets:
        pred = [n for n in G.predecessors(node)]
        for p in pred:
            npfile.write(" "+ str(p) +" "+ str(noisyParameters[p]))
    npfile.write("\n")
npfile.close()


for s in range(0,sampleSize*2):   
    if debug:
        print("  Sample No "+str(s+1))
        print (pred)

    sample = []
    for i in range(0, numberOfVariables):
        sample.append(0) 
    
    for node in nx.topological_sort(G):
        pred = [n for n in G.predecessors(node)]       
    
        if pred == []:
            sample[node] = np.random.randint(0,2)
            #noiseS =  flipWithNoise(sampleNoise, sample[node])
            #if sample[node]!= noiseS:
            #    print("flipped")
            #sample[node] = noiseS


            if debug:
                print("    Sampling", node, "with parent set", str(pred)+":", sample[node])
        else:
            threshold = 0
            if node in variablesWithNoisyParentSets:
                product = 1
                for p in pred:
                    if sample[p] == 1:
                        product = product * noisyParameters[p]
                        
            else:             
                product = 1
                for p in pred:
                    if sample[p] == 1:
                        product = 0
                            
                        
                    #if uniform(0,1) > noisyParameters[node]:
                    #    sample[node] = value
                #if sample[p] == 0:
                    #if uniform(0,1) < noisyParameters[node]:
                    #    sample[node] = 1
            if uniform(0,1) <= 1-product:
                sample[node] = 1
            else:
                sample[node] = 0
            #noiseS =  flipWithNoise(sampleNoise, sample[node])
            #if sample[node]!= noiseS:
            #    print("flipped")
            #sample[node] = noiseS
            if debug: 
                print("    Sampling", node, "with parent set", str(pred)+":", sample[node])
    if s <  sampleSize :         
        fcsv.write(str(sample).strip('[]') + "\n")
        scpy = sample
        y = scpy.pop(scpy[0])
        scpy.append(y)
        fcsv2.write(str(scpy).strip('[]').replace(" ","") + "\n")

    else :
        fevidence.write(str(sample).strip('[]') +"\n")



variableValues = ["no", "yes"]
fbif.write("network unknown {")
fbif.write("}\n")
for i in range(0, numberOfVariables):
    fbif.write("variable v"+str(i)+" {")
    fbif.write("    type discrete [2] { no, yes };")
    fbif.write("}\n")
fbif.close()

sample = []
for i in range(0, numberOfVariables):
    sample.append(0)     

for node in nx.topological_sort(G):
    pred = [n for n in G.predecessors(node)]       
        
    if pred == []:
        fbif = open(outfile + ".bif", "a")
        fbif.write("probability ( v"+str(node)+" ) {")
        fbif.close()
        cpd = NoisyORCPD(noisyParameters, node, pred)
        cpd.printCPD(outfile +".bif")
        #print("    table 0.5, 0.5;")
        fbif = open(outfile + ".bif", "a")
        fbif.write("} \n")
        fbif.close()
    else:
        string = "v"+str(node)+" | "
        first = True
        for p in pred:
            if first:
                first = False
            else:
                string = string + ", "
            string = string + "v"+str(p)

        fbif = open(outfile + ".bif", "a")
        fbif.write("probability ( "+string+" ) {") 
        fbif.close()
        cpd = NoisyORCPD(noisyParameters, node, pred)
        cpd.printCPD(outfile + ".bif")
        fbif = open(outfile + ".bif", "a")
        fbif.write("} \n")  
        fbif.close()


fcsv.close()
fcsv2.close()
fevidence.close()

#run python3 generateNoisyOR.py numVar sampleSize prefixName


