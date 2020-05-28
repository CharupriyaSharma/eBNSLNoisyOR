import numpy as np
from numpy import genfromtxt

s = [3,4,5,6,7,8]
N = [100, 500, 1000]
print("median \\")
for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("ERR_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.median(a),2))
        sr +=" & "
        a = genfromtxt("ERR_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.median(a),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)
print("75th percentile \\\\")
for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("ERR_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,75),2))
        sr +=" & "
        a = genfromtxt("ERR_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,75),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)


print("90th percentile \\\\")
for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("ERR_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,90),2))
        sr +=" & "
        a = genfromtxt("ERR_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,90),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)



print("Conditional KL \n")

for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("CKL_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.median(a),2))
        sr +=" & "
        a = genfromtxt("CKL_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.median(a),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)



print("75th percentile \\\\")

for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("CKL_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,75),2))
        sr +=" & "
        a = genfromtxt("CKL_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,75),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)
print("90th percentile \\\\")

for i in s :
    sr=str(i-1) + " & "
    for j in N:
        a = genfromtxt("CKL_GD_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,90),2))
        sr +=" & "
        a = genfromtxt("CKL_EM_"+str(i)+"_"+str(j))
        sr += str(np.round(np.percentile(a,90),2))
        if j < 1000:
            sr +=" & "

        else:
            sr +=" \\\\"
    print(sr)


