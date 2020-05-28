import math
import random
import itertools
import sys

def printInferenceQueries(data, network, nodes):
    
    nodelist = list(range(0,nodes))
    
    f = open(data+".evidence.csv", "r")
    lines = f.readlines()

    outputfile = "query_"+ data + ".R"
    f2 = open(outputfile, "w+")
    f2.write("library(bnlearn)\n")
    f2.write("net = read.bif(\""+ network + "\")\n" )
    f2.write("sink(\"" +  "query_"+ data  + ".csv\")\n")
    

    
    for line in lines:
        nodelistcopy = nodelist.copy()
        infnodelist = []
        infnodelen = math.ceil(nodes/10)
        
        cpquerystr = []
        for i in range(0,2**infnodelen):
            cpquerystr.append( "cpquery(net, (")
        
        testvals = list(itertools.product([0, 1], repeat=infnodelen))
        

        for i in range(0,infnodelen):
            n = random.randint(0,len(nodelistcopy)-1)
            infnodelist.append(nodelistcopy[n])
            nodelistcopy.remove(nodelistcopy[n])

        for i in range(0,2**infnodelen):
            for j in range(0,infnodelen):
                if testvals[i][j] == 0:
                    cpquerystr[i] += "v" + str(infnodelist[j])+  " == \"no\""
                else:
                    cpquerystr[i] += "v" + str(infnodelist[j])+ " == \"yes\""

                    if j < infnodelen-1  :
                        cpquerystr[i] += " & "    
                         
        for i in range(0,2**infnodelen):    
            cpquerystr[i] += "),("
        
        valsfromsampling = [int(s) for s in line.strip().split(", ") if s.isdigit()]
        for i in range(0,len(nodelistcopy)):
            for j in range(0,2**infnodelen):
                if valsfromsampling[nodelistcopy[i]] == 0:
                    cpquerystr[j] += "v" + str(nodelistcopy[i])+ " == \"no\""
                else:
                    cpquerystr[j] += "v" + str(nodelistcopy[i])+ " == \"yes\""

                if i < len(nodelistcopy)-1 :
                    cpquerystr[j] += " & "
        
        for i in range(0,2**infnodelen):    
            cpquerystr[i] += "))\n"
        
        for i in range(0,2**infnodelen):
            f2.write(cpquerystr[i])
                
    f2.write("sink()\n")
    
    f.close()
    f2.close()

    
printInferenceQueries(sys.argv[1], sys.argv[2], int(sys.argv[3]))


