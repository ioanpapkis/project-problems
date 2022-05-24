# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:07:24 2020

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
γ = 9*10**(-7) 
r0 = y + k
δ = 10**(-7)

w = int(input('enter the number of weeks:'))
time = []
for i in range(w+1):
    time.append(i)
    
    
  
for i in range(w):
    S1 = S[i] - a*S[i]*I[i] + γ*I[i]
    I1 = I[i] + a*S[i]*I[i] - γ*I[i] 
    S.append(S1)
    I.append(I1)

 
    #συναρτηση για να υπολογίσουμε την κορυφωση της εξαπλωσης της ασθενειας
def maxm(L):
    s = L[0]
    k = 0
    for i in range(len(L)):
        if s < L[i]:
            s = L[i]
            k = i
    return (s,k)     
       


for i in range(w+1):
    print('week',i,'S=',int(S[i]),'I=',int(I[i]))
 
print('In a span of ',w,'weeks, week number',maxm(I)[1],'is the peak week of the disease with a total of',int(maxm(I)[0]),'infected people')
    


#p.plot(time,S,'r-')
