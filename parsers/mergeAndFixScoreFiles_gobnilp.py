import sys
from math import log
bicfile = sys.argv[1] + "." + sys.argv[2] 
freg = open(bicfile, "r")
fmerge = open(sys.argv[1] + "." + sys.argv[2] + ".merged", "w+")
N = int(sys.argv[3])
#get var count

nstr = freg.readline()
n= int(nstr)

fmerge.write(nstr)

for i in range(n):

    scoresNO = []
    scoresREG = []
    parentsNO = []
    parentsREG = []

    #read scores from noisy-or file for ithvariable 
    fno = open(sys.argv[1] + "_" + str(i) +"_" + sys.argv[2])
    f2 = fno.readlines()
    
    for line in f2 : 
        items = [float(it) for it in line.split()]
        
        if len([int(p) for p in items[2:]]) > 1 :
                parsize = len(items)-2
                newpenalty = (log(N)/2*(2^parsize)) - parsize/2 *log(N)
                scoresNO.append(items[0] + newpenalty)
                parentsNO.append([int(p) for p in items[2:]])
    fno.close()
    
    #get scores for ith variable  
    scorecountstr = freg.readline()
    scorecountitems = [int(it) for it in scorecountstr.split()]
    scorecount = scorecountitems[1]

    for j in range(scorecount):
        line = freg.readline()

        items = [float(it) for it in line.split()]
        parentset = [int(p) for p in items[2:]]
        score = items[0]

        if parentset in parentsNO :
            conflictid = parentsNO.index(parentset)
            print (scoresNO[conflictid] -  score, parentset, parentsNO[conflictid])
            if score >=  scoresNO[conflictid] :
                parentsNO.pop(conflictid)
                scoresNO.pop(conflictid)
                scoresREG.append(score)
                parentsREG.append(parentset)
        else:
            scoresREG.append(score)
            parentsREG.append(parentset)

                
        for ps in parentsNO:
                psindex = parentsNO.index(ps)
                if set(parentset).issubset(set(ps)) and score >= scoresNO[psindex]  :
                    parentsNO.pop(psindex)
                    scoresNO.pop(psindex)
                
    #write merged scores to file
    totalscorecount = len(scoresREG) + len(scoresNO)
    fmerge.write( str(i) + " " + str(totalscorecount) + "\n")

    for k in range(len(scoresNO)) :
        fmerge.write(str(scoresNO[k])+ " " + str(len(parentsNO[k])) + " " +  " ".join(str(it) for it in parentsNO[k]) + "\n")

    for k in range(len(scoresREG)) :
        fmerge.write(str(scoresREG[k])+ " "  + str(len(parentsREG[k])) + " " + " ".join(str(it) for it in parentsREG[k])+ "\n")


freg.close()
fmerge.close()
