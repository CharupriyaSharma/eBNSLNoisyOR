import sys

nps=[]
noise=[]
ds = sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]
ferr = open(ds+"_noise", "r")
lines=ferr.readlines()
for l in lines:
    if l[0] =="0":
        np = [float(i) for i in l.split()]
        nps=np[2::2]
ferr1 = open(ds+"_0_3_noise", "r")
lines=ferr1.readlines()
for l in lines:
    if l[0] =="0":
        np = [float(i) for i in l.split()]
        noise=np[2::2]

err = 0
#print(nps)
#print(noise)
for i in range(0,len(nps)):
    err += abs(nps[i] - noise[i])/nps[i]
#print(err)


f=open("ERR_EM_"+ sys.argv[2] + "_" + sys.argv[3], "a+")
f.write(str(err/len(nps)))
f.write("\n")
f.close()
