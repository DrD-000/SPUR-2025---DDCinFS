# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:34:05 2025

@author: Admin
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
rng=np.random.default_rng()
fig, R=plt.subplots()

a=1                     # - Constant
n=200                   # - #Sites
m=0                     # - Disorder Coefficient
d=0.5                   # - Wave Packet: Width
o=100                   # - Wave Packet: Origin
t=np.arange(10,61,10)   # - Time intervals
K=10                    # - #Trials (Not yet done!)


N=list(range(n))
H=np.random.rand(n,n)

for i in range(n):
    for j in range (n):
        if j==i:
            H[i][j]=2*a+m*rng.random()
        elif j==i+1:
            H[i][j]=-a
        elif j==i-1:
            H[i][j]=-a
        else:
            H[i][j]=0
H[0][n-1]=-a
H[n-1][0]=-a
H=H.astype(complex)

E,v=eig(H)
V=np.transpose(v)
    
p=np.random.rand(n)
for i in range(n):
    p[i]=np.sqrt(d)*(np.e**(-(d**2)*((i-o)**2)))
p=p.astype(complex)

c=np.linalg.solve(v,p)
C=np.reshape(c,(n,))

for k in t:
    
    q=np.random.rand(n,n)
    q=q.astype(complex)

    for i in range(n):
        for j in range (n):
            q[i,j]=C[i]*(np.exp(-E[i]*1j*k))*V[i,j]
    q=np.transpose(q)

    Q=np.random.rand(n)
    Q=Q.astype(complex)

    for i in range(n):
        Q[i]=np.sum(q[i])

    P=abs(Q)**2
    
    R.plot(N,P,label="t="+str(k))
plt.title("n="+str(n)+" m="+str(m))
plt.legend()
plt.show()

