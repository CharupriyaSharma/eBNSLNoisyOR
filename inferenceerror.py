import sys
import numpy as np
#python3 inferenceerror.py base noisy-or noisy-or-last bic

base = np.loadtxt(sys.argv[1], delimiter="\n", dtype=float)
base[base == 0] = 1
b0=base[0:len(base):2]
b1=base[1:len(base):2]

no = np.loadtxt(sys.argv[2], delimiter="\n", dtype=float)
n0=no[1:len(base):2]
n1=no[0:len(base):2]

noe = np.loadtxt(sys.argv[3], delimiter="\n", dtype=float)
ne0=noe[1:len(base):2]
ne1=noe[0:len(base):2]


cpd = np.loadtxt(sys.argv[4], delimiter="\n", dtype=float)
c0=cpd[0:len(base):2]
c1=cpd[1:len(base):2]

nb0 = np.abs(np.divide(b0-n0, b0))
nb1 = np.abs(np.divide(b1-n1, b1))
nbs = np.concatenate((nb0, nb1), axis=0)

neb0 = np.abs(np.divide(b0-ne0, b0))
neb1 = np.abs(np.divide(b1-ne1, b1))
nebs = np.concatenate((neb0, neb1), axis=0)


cb0 = np.abs(np.divide(b0-c0, b0))
cb1 = np.abs(np.divide(b1-c1, b1))
cbs = np.concatenate((cb0, cb1), axis=0)

print(f"{np.median(nbs):.4f}")
print(f"{np.median(nebs):.4f}")
print(f"{np.median(cbs):.4f}")



print(f"{np.mean(nbs):.4f}")
print(f"{np.mean(nebs):.4f}")
print(f"{np.mean(cbs):.4f}")


