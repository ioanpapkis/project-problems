#Math 5511 Ιωάννης Παπάκης

import numpy as np

def Powermethod(A):
    kmax = 100 # max bhmata
    D = [1]*kmax # arxikes listes opou kratane ta dk+1/dk stoixeia 
    L = [1]*kmax # 
    u0 = np.random.rand(A.shape[0]) # tyxaio dianysma ston Rn 
    absu0= np.linalg.norm(u0, ord=1) # l1 norma tou dianysmatos 
    x0 = u0/absu0 #arxiko dianysma 
    tol = 1.e-6 #error 
    k = 0
    dk = 1 
    while k < kmax and dk > tol:
        u1 = np.matmul(A,x0)
        absu1 = np.linalg.norm(u1,ord=1)
        xk = u1/absu1 
        dk = np.linalg.norm((xk-x0),ord=1)
        D[k]=dk 
        k +=1
        x0 = xk
    l = np.matmul(xk.T,np.matmul(A,xk))/np.matmul(xk.T,xk)
    for i in range(len(D)-1): #ftiaxnoyme ton logo dk+1/dk 
        L[i]=D[i+1]/D[i] 
        
    return l,xk,L[10] #epistrefei kyriarxo idiodianusma/idiotimh kai ton logo dk+1/dk

#πχ εχουμε τους πινακες
A1 = np.array([[1,0,4],[0,2,4],[0,0,4]]) #idiotimes l1=4,l2=2,l3=1 l2/l1=1/2

#triad[-1,2,-1] για n = 4
A2 = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
eigvalue=list(2-2*np.cos(np.pi*i/(A2.shape[0]+1)) for i in range(1,A2.shape[0]+1))
eigvalue.sort(reverse=True) # idiotimes tou tridiagwniou pinaka
S = eigvalue[1]/eigvalue[0] # logos idiotimwn

print('Για τον πίνακα Α1 έχουμε λ1:',Powermethod(A1)[0],'x1:',Powermethod(A1)[1],'λογος dk+1/dk:',Powermethod(A1)[2],'λογος λ2/λ1: 0.5','\n')
print('Για τον πίνακα Α2 έχουμε λ1:',Powermethod(A2)[0],'x1:',Powermethod(A2)[1],'λογος dk+1/dk:',Powermethod(A2)[2],'λογος λ2/λ1:',S)

print(S)


