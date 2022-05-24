#Ιωάννης Παπάκης math5511




import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline




def data(x,y,k):
    train = pd.read_table(x,delim_whitespace=True)
    test = pd.read_table(y,delim_whitespace=True) 
    
    
    for i in range(2,k+1):
        train.insert(int(len(train.columns)),'i'*i,train.iloc[:,0]**i)
        test.insert(int(len(test.columns)),'i'*i,test.iloc[:,0]**i)
    
    for i in range(train.shape[1]):
        train.iloc[:,i] = train.iloc[:,i]/max(train.iloc[:,i])
        test.iloc[:,i] = test.iloc[:,i]/max(test.iloc[:,i])
    
    y_target =  train.iloc[:,1]
    y_target_test  = test.iloc[:,1]
    train = train.drop('y',axis=1)
    test  = test.drop('y',axis=1)
    
    train.insert(0,'intercept', [1]*train.shape[0])
    test.insert(0,'intercept',[1]*test.shape[0])
    
    return  train, test , y_target, y_target_test



train = data('f_train.txt','f_test.txt',3)[0]
test  = data('f_train.txt','f_test.txt',3)[1]
y_target = data('f_train.txt','f_test.txt',3)[2]
y_target_test = data('f_train.txt','f_test.txt',3)[3]
n = train.shape[0] # number of data in each column
m = train.shape[1] # number of features   



#function for gradient derivative        
def  grad(x,m,a,s,n):
    errors = np.subtract(a,s)
    sumerror = 1/n*x.transpose().dot(errors)
    return sumerror



def GradientDescent(x,y):
    global J
    global k
    global s
    global e
    global d   
    global th
    global y_pred
    train = data(x,y,3)[0]
    test  = data(x,y,3)[1]
    y_target = data(x,y,3)[2]
    y_target_test = data(x,y,3)[3]
    n = train.shape[0]
    m = train.shape[1]
    e = 1.e-4 #errors of tolerance  
    d = 1.e-4
    k = 0 #steps
    s = 0.1 #learning rate 
    jth= 0  
    y_pred = np.zeros(n)  
    jth = 1
    therror= 1 
    th2 = np.zeros(m) 
    difj=np.zeros(m) # derivative of cost function 
    th = np.zeros(m) #Regression coefficients
    J = []
    #h = np.empty(n)
    while abs(jth) >= e and therror >= d:
        y_pred = train.dot(th)
        difj=grad(train,m,y_pred,y_target,n)
        th2=th - s*difj   
        jth = 2/n * sum(np.square((y_pred - y_target)))
        J.append(jth)
        therror = np.linalg.norm(th2 - th,ord = 1) 
        th = th2
        k+=1   
    err2 = np.linalg.norm(test.dot(th)-y_target_test, ord=2)
    
    
    return err2,th
      


GradientDescent('f_train.txt','f_test.txt')

#plot h(x^)
hcost = train.dot(th)

plt.scatter(train.iloc[:,1],y_target,color = 'blue')
plt.plot(train.iloc[:,1],hcost,'ro')
plt.xlabel('xi')
plt.ylabel('yi')
plt.show()






# Normal equations 
def Normaleq(x,y):
    test = data(x,y,3)[1]
    train = data(x,y,3)[0]
    y_target = data(x,y,3)[2]
    y_target_test = data(x,y,3)[3]
    
    theta = np.linalg.inv(train.T.dot(train)).dot(train.T).dot(y_target)
    y_predict = train.dot(theta)
    jth = 2/n * sum(np.square((y_predict - y_target)))
    err2 = np.linalg.norm(test.dot(theta)-y_target_test, ord=2)
    return err2,theta




print(GradientDescent('f_train.txt','f_test.txt'))
print(Normaleq('f_train.txt','f_test.txt'))


def GradientDescent2(x,k):
    global theta
    df = data(x,'f_test.txt',k)[0]
    y_target = data(x,'f_test.txt',k)[2]
    n = df.shape[0]
    m = df.shape[1]
    e = 1.e-4 #errors of tolerance  
    d = 1.e-3
    k = 0 #steps
    s = 0.001 #learning rate 
    jth= 0  
    y_predict = np.zeros(n)  
    jth = 1
    therror= 1 
    th2 = np.zeros(m) 
    difj=np.zeros(m) # derivative of cost function 
    theta = np.zeros(m) #Regression coefficients
    J = []
    #h = np.empty(n)
    while abs(jth) >= e and therror >= d:
        y_predict = df.dot(theta)
        difj=grad(df,m,y_predict,y_target,n)
        
        th2=theta - s*difj   
        jth = 2/n * sum(np.square((y_predict - y_target)))
        J.append(jth)
        therror = np.linalg.norm(th2 - theta,ord = 1) 
        theta = th2
        k+=1
    return theta


legend = ['k=3','k=5','k=10','k=20']


data3 =  data('f_small.txt','f_test.txt',3)[0]
data5 =  data('f_small.txt','f_test.txt',5)[0]
data10 =  data('f_small.txt','f_test.txt',10)[0]
data20 = data('f_small.txt','f_test.txt',20)[0]

theta3 = GradientDescent2('f_small.txt',3)
theta5 = GradientDescent2('f_small.txt',5)
theta10 = GradientDescent2('f_small.txt',10)
theta20 =  GradientDescent2('f_small.txt',20)


h3 = data3.dot(theta3)
h5 = data5.dot(theta5)
h10 = data10.dot(theta10)
h20 = data20.dot(theta20)




plt.plot(data3.iloc[:,1],h3,'ro')
plt.plot(data3.iloc[:,1],h5,'go')
plt.plot(data3.iloc[:,1],h10,'bo') 
plt.plot(data3.iloc[:,1],h20,'mo')
plt.ylabel('h(θ)')
plt.legend([legend[0],legend[1],legend[2],legend[3]])
plt.show()
















    



