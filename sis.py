# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:09:55 2020

@author: johnp
"""
import matplotlib.pyplot as p

x=5
y=5
z=1
k=1

I0 = 1000
S0 = 10000
N = I0  + S0 

# αρχικες λιστες καθε πληθυσμου
S = []
I = []
R = []
D = []
S.append(S0)
I.append(I0)



a = 54*10**(-6)
γ = 9*10**(-2) 
r0 = y + k
δ = 10**(-2)

w = int(input('enter the number of weeks:'))
time = []
for i in range(w+1):
    time.append(i)
    
    
  
for i in range(w):
    S1 = S[i] - a*S[i]*I[i] + γ*I[i]
    I1 = I[i] + a*S[i]*I[i] - γ*I[i] 
    S.append(S1)
    I.append(I1)



p.plot(time,S,'r-')
p.plot(time,I,'g-')
p.legend(['S','I'])
p.xlabel('Time(in weeks)')
p.ylabel('Population(N)')
p.show()






