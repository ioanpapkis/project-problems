# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:36:12 2020

@author: johnp
"""

import numpy as np 
import matplotlib.pyplot as p



x=5
y=5
z=1
k=1

I0 = 1000
R0 = 0
S0 = 10000
D0 = 0
N = I0 + R0 + S0 + D0

# αρχικες λιστες καθε πληθυσμου
S = []
I = []
R = []
D = []
S.append(S0)
I.append(I0)
R.append(R0)
D.append(D0)


a = 54*10**(-6)
γ = 9*10**(-2) 
r0 = y + k
δ = 10**(-2)

w = int(input('enter the number of weeks:'))
time = []
for i in range(w+1):
    time.append(i)
    
    
  
for i in range(w):
    S1 = S[i] - a*S[i]*I[i]
    I1 = I[i] + a*S[i]*I[i] - γ*I[i] -δ*I[i]
    R1 = R[i] + γ*I[i]
    D1 = D[i] + δ*I[i]
    S.append(S1)
    I.append(I1)
    R.append(R1)
    D.append(D1)
 
    


p.plot(time,S,'r-')
p.plot(time,I,'g-')
p.plot(time,R,'b-')
p.plot(time,D,'k')
p.legend(['I'])
p.xlabel('Time(in weeks)')
p.ylabel('Population(N)')
p.show()

    
    
    
    
    
    
    
    
    
    