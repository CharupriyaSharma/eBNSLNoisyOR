import sys
import os

noisefile = sys.argv[2]
freg = open(noisefile, "r")
fmerge = open(sys.argv[1] + "_noise_compare", "w+")

#get var count

n= int(sys.argv[3])
realNoise = [0]*n
computedNoise = [0]*n
error = [0]*n
for i in range(n):

    
    
    #get scores for ith variable  
    item = freg.readline()
    items = [float(it) for it in item.split()]
    scorecount = 0
    if len(items)>1:
        scorecount = int(len(items)/2)
    print(items)
    print(scorecount)
    if scorecount > 0 :
        for j in range(scorecount):
            print("scoring node : ")
            print(items[0])
            realNoise[int(items[2*j+1])] =  items[2*j+2]
          #  print(str(int(items[2*j+1])) + ":"  +str(items[2*j+2]))
                #read moise from noisy-or file for ithvariable
            nifile = sys.argv[1] + "_" + str(int(items[0])) + "_20_noise" 
            print("opening " + nifile)
            fno = open(nifile)
            f2=""
            if os.stat(nifile).st_size > 0 : 
                f2 = fno.readlines()
        
            for line in f2 :
                print ("computed " + line)
                items1 = [float(it) for it in line.split()]
                if int(items1[0]) > 0 :
                    paracount = int(items1[0])
                    for k in range(paracount):
                        print(str(int(items1[k+1])) + ":" + str(items1[paracount+1+ k]))
                        print(int(items1[k+1]))
                        currenterr = abs(computedNoise[int(items1[k+1])] - realNoise[int(items1[k+1])])
                        newerr = abs(items1[paracount+1+ k] - realNoise[int(items1[k+1])])
                        print(abs(newerr < currenterr))
                        if computedNoise[int(items1[k+1])] == 0  or abs(newerr < currenterr):
                            computedNoise[int(items1[k+1])] =  items1[paracount+1+ k]
    
            fno.close()


                
    #write merged scores to file
err = 0
errctr = 0
for i in range(n):
    
    if realNoise[i] >0 :
        err1 = abs(realNoise[i]-computedNoise[i])
        errctr+=1
        err+=err1
        error[i] = err1

    fmerge.write(str(i) + " " + str(realNoise[i]) + " " + str(computedNoise[i])+ " " + str(error[i]) + "\n")
if errctr > 0 :
    fmerge.write("error : " + str(err/errctr)+ "\n")

freg.close()
fmerge.close()

er =open("errors", "a+")
er.write(sys.argv[1] + " :" + str(err/errctr) + "\n")
er.close()
