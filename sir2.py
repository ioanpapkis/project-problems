# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:16:13 2020

@author: johnp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:36:12 2020

@author: johnp
"""


x=5
y=5
z=1
k=1

#αρχικες συνθηκες
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

#παραμετροι μοντελου
a = 54*10**(-6)
γ = 9*10**(-2) 
r0 = y + k
δ = 10**(-2)

w = int(input('enter the number of weeks:'))
time = []
for i in range(w+1):
    time.append(i)
    
    
#λυση συστηματος
for i in range(w):
    S1 = S[i] - a*S[i]*I[i]
    I1 = I[i] + a*S[i]*I[i] - γ*I[i] -δ*I[i]
    R1 = R[i] + γ*I[i]
    D1 = D[i] + δ*I[i]
    S.append(S1)
    I.append(I1)
    R.append(R1)
    D.append(D1)
 
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
    print('week',i,'S=',int(S[i]),'I=',int(I[i]),'R=',int(R[i]),'D=',int(D[i]))
 
print('In a span of ',w,'weeks, week number',maxm(I)[1],'is the peak week of the disease with a total of',int(maxm(I)[0]),'infected people')