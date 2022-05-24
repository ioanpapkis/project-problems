#Ιωάννης Παπάκης math5511




import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LogisticRegression



def sigmoid(x):
    return 1/(1 + np.exp(-x))




def data(x,y):
    train = pd.read_table(x,delim_whitespace=True)
    test = pd.read_table(y,delim_whitespace=True) 
    
    for i in range(train.shape[1]):
        train.iloc[:,i] = train.iloc[:,i]/max(train.iloc[:,i])
        test.iloc[:,i] = test.iloc[:,i]/max(test.iloc[:,i])
        
    y_target =  train.iloc[:,2]
    y_target_test  = test.iloc[:,2]
    train = train.drop('y',axis=1)
    test  = test.drop('y',axis=1)
    
    train.insert(0,'intercept', [1]*train.shape[0])
    test.insert(0,'intercept',[1]*test.shape[0])
    
    return train,test ,y_target , y_target_test



train_set = data('set1_train.txt','set1_test.txt')[0]
train_set2 = data('set2_train.txt','set2_test.txt')[0]
test_set = data('set1_train.txt','set1_test.txt')[1]
test_set2 = data('set2_train.txt','set2_test.txt')[1]

y_target = data('set1_train.txt','set1_test.txt')[2]
y_target2 = data('set2_train.txt','set2_test.txt')[2]
y_target_test = data('set1_train.txt','set1_test.txt')[3]
y_target_test2 = data('set2_train.txt','set2_test.txt')[3]



# grad of l(θ)
def grad(x,y,s):
    sub = y - sigmoid(x.dot(s))
    g = sub.dot(x)
    return g


    
# Hessian matrix 
def Hessian(x,y): 
    s = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        s[i]= sigmoid(x.iloc[i].dot(y))*(1-sigmoid(x.iloc[i].dot(y)))
    h = -x.T.dot(np.diag(s)).dot(x)
    return h 
            


def Logregression(x,y,test):
    thet = np.ones(x.shape[1]) 
    th2 =  np.zeros(x.shape[1]) 
    e = 1.e-4 
    k = 0 
    error = 1
    while error >= e: 
        hess = np.linalg.inv(Hessian(x,thet))
        g = grad(x,y,thet)   
        th2 = thet - hess.dot(g) 
        
        error = np.linalg.norm(th2-thet,ord=1)
        thet = th2
        k +=1 
    
    predict1 = np.zeros(test.shape[0])
    predict0 = np.zeros(test.shape[0])
    
    for i in range(test_set.shape[0]):
        predict1[i]=sigmoid(test.iloc[i].dot(thet))
        predict0[i]=1-sigmoid(test.iloc[i].dot(thet))
        
    return thet,predict1,predict0

print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 1','y=1',Logregression(train_set,y_target,test_set)[1],'\n')
print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 1','y=0',Logregression(train_set,y_target,test_set)[2],'\n')
print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 2','y=1',Logregression(train_set2,y_target2,test_set2)[1],'\n')
print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 2','y=0',Logregression(train_set2,y_target2,test_set2)[2])


# sklearn 
clf = LogisticRegression(solver='newton-cg')
clf.fit(train_set,y_target)

clf2 = LogisticRegression(solver='newton-cg')
clf2.fit(train_set2,y_target2)

s1 = clf.coef_
s2 = clf2.coef_



print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 1 ',clf.predict_proba(train_set),'\n') #πρωτη στηλη ειναι πιθ για y=0 και δευτερη για y=1

print('προβλεψεις πιθανοτητας για το συνολο ελεγχου 2 ',clf.predict_proba(train_set2))


#plot data 
def plotreg(x,y,theta):
    theta.tolist()
    th0 = theta[0]
    th1 = theta[1]
    th2 = theta[2]
    y00 = [] # stoixeia toy x1 me y=0
    y01 = [] # stoixeia toy x2 me y=0
    y10 = [] # stoixeia toy x1 me y=1
    y11 = [] #stoixeia  toy x2 me y=1 
    for i in range(x.shape[0]):
        if y[i] == 0:
            y00.append(x.iloc[i][1])
            y01.append(x.iloc[i][2])
        if y[i] == 1:
            y10.append(x.iloc[i][1])
            y11.append(x.iloc[i][2])
    x_ = y01+y00
    y_=[]
    for i in range(len(x_)):
        y_.append(-(th1/th2)*x_[i]-(th0/th2))
            
    plt.scatter(y00,y01,s=8) #plot τα δεδομενα για y=0
    plt.scatter(y10,y11,s=8) #plot τα δεδομενα για y=1
    plt.plot(x_,y_,'k--') #plot το συνορο αποφασης
    plt.xlabel('x1')
    plt.ylabel('x2')  
    plt.legend(['Decision boundary','y=0','y=1']) 
    plt.show()
    

# plot toy train set 1 kai train set 2
print('plot για train set 1')    
print(plotreg(train_set,y_target,Logregression(train_set,y_target,test_set)[0]))
print('plot για train set 2')   
print(plotreg(train_set2,y_target2,Logregression(train_set2,y_target2,test_set)[0]))



#### plot graphs with sklearn regression coef  train sets
#print(plotreg(train_set2,y_target2,s2[0]))
#print(plotreg(train_set,y_target,s1[0]))



### plot graph on test set  and test set 2
print('plot για test set 1')   
print(plotreg(test_set,y_target_test,Logregression(test_set,y_target_test,test_set)[0]))
print('plot για test set 2')
print(plotreg(test_set2,y_target_test2,Logregression(test_set2,y_target_test2,test_set)[0]))













