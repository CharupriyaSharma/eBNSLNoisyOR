import sys
from math import log
bicfile = sys.argv[1] + ".bic.opt" 
freg = open(bicfile, "r")
fmerge = open(sys.argv[1]  + ".bic", "w+")
#get var count

nstr = freg.readline()
n= int(nstr)

fmerge.write(nstr)

for i in range(n):

    scoresREG = []
    parentsREG = []

    
    #get scores for ith variable  
    scorecountstr = freg.readline()
    scorecountitems = [int(it) for it in scorecountstr.split()]
    scorecount = scorecountitems[1]

    for j in range(scorecount):
        line = freg.readline()

        items = [float(it) for it in line.split()]
        parentset = [int(p) for p in items[1:]]
        score = items[0]

        scoresREG.append(score)
        parentsREG.append(parentset)
                
                
    #write merged scores to file
    fmerge.write( str(i) + " " + str(scorecount) + "\n")

    for k in range(len(scoresREG)) :
        fmerge.write(str(scoresREG[k])+ " "  + str(len(parentsREG[k])) + " " + " ".join(str(it) for it in parentsREG[k])+ "\n")


freg.close()
fmerge.close()
