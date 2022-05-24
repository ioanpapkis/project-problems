#Ιωάννης Παπάκης math5511



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


#normalise the data
def normalise(x,y):
    data_train = pd.read_table(x,delim_whitespace=True)
    data_test = pd.read_table(y,delim_whitespace=True)
    
    for i in range(data_train.shape[1]):
        mean_train = data_train.iloc[:,i].mean()
        std_train = data_train.iloc[:,i].std()
        
        mean_test = data_test.iloc[:,i].mean()
        std_test = data_test.iloc[:,i].std()
        
        data_train.iloc[:,i]= (data_train.iloc[:,i] - mean_train)/std_train
        data_test.iloc[:,i]=(data_test.iloc[:,i]-mean_test)/std_test
        
    return data_train,data_test 
    
  
df_train = normalise('car_train.txt','car_test.txt')[0]
df_test = normalise('car_train.txt','car_test.txt')[1] 
n = df_train.shape[0] # number of data in each column
m = df_train.shape[1] # number of features   


y_target = df_train.iloc[:, 4]
y_target_test  = df_test.iloc[:,4]


df_train = df_train.drop('km/l',axis=1)
df_test = df_test.drop('km/l',axis=1)


df_train.insert(0,'intercept', [1]*df_train.shape[0])
df_test.insert(0,'intercept',[1]*df_test.shape[0])


#function for gradient derivative        
def  grad(x,m,a,s,n):
    errors = np.subtract(a,s)
    sum_delta = 1/n*x.T.dot(errors)
    return sum_delta



def GradientDescent(x,y):
    global J
    global k
    global s
    global e
    global d
    global th
    e = 1.e-4 #errors of tolerance  
    d = 1.e-5
    k = 0 #steps
    s = 0.15 #learning rate 
    jth= 0  
    y_pred = np.zeros(n)  
    jth = 1
    therror= 1 
    th2 = np.zeros(m) 
    difj=np.zeros(m) # derivative of cost function 
    th = np.zeros(m) #Regression coefficients
    J = []
    while abs(jth) >= e and therror >= d:
        y_pred = x.dot(th)
        difj=grad(df_train,m,y_pred,y_target,n)
        th2=th - s*difj   
        jth = 2/n * sum(np.square((y_pred - y_target)))
        J.append(jth)
        
        therror = np.linalg.norm(th2 - th,ord = 1) 
        th = th2
        k+=1   
    err2 = np.linalg.norm(y.dot(th)-y_target_test, ord=2)
    
    return th,err2,k,jth  
      

#plot cost function 
plt.plot(list(range(GradientDescent(df_train,df_test)[2])),J,'b')
plt.grid()
plt.legend(['J(θ)'])
plt.xlabel('Number of steps')
plt.ylabel('J(θ)')


# Normal equations 
def Normaleq(x,y):
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    y_predict = x.dot(theta)
    jth = 2/n * sum(np.square((y_predict - y)))
    err2 = np.linalg.norm(df_test.dot(theta)-y_target_test, ord=2)
    return theta,err2,jth


   
    
#Linear regression with sk learn    
MLR = LinearRegression()
MLR.fit(df_train,y_target)



predictions = MLR.predict(df_test)
error2 = np.linalg.norm(predictions-y_target_test,ord=2)


coef= MLR.coef_
coef[0]=MLR.intercept_



print('Steepest descent:','theta=',GradientDescent(df_train,df_test)[0],'Steps=',GradientDescent(df_train,df_test)[2],'\n')
print('J(Θ)=',GradientDescent(df_train,df_test)[3],'learning rate=',s,'error2=',GradientDescent(df_train,df_test)[1],'\n')
print('ε=',e,'δ=',d,'\n')
print('Normal equations:','theta=',Normaleq(df_train,y_target)[0],'jth=',Normaleq(df_train,y_target)[2],'error2=',Normaleq(df_train,y_target)[1],'\n')
print('sklearn LinearRegression:','theta=',coef,'error2=',error2)    
    



    
    
    