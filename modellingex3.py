# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:18:27 2020

@author: johnp
"""

import numpy as np
import matplotlib.pyplot as plt

#AM
x = 5
y = 5
z = 1
k = 1

q = (x+y+z+k)*10**(-10)
m1 = (x+z)*10**(-9)
Ex=x
Ey=y
B =z
vc= (q*B)/m1


# initial conditions
t0 = float(input('enter the initial time:'))
x0 = float(input('enter the initial x coordinate of the trajectory: '))
y0 = float(input('enter the initial y coordinate of the trajectory: '))
ux0 = float(input('enter the initial x coordinate of the velocity: '))
uy0 = float(input('enter the initial y coordinate of the velocity: '))
d = np.arctan(-(uy0+(Ex/B))/(ux0-(Ey/B)))-vc*t0 #Αρχικη φαση προβληματος
u0 = (ux0-(Ey/B))/np.cos(vc*t0+d) 
C3 = x0 -(u0/vc)*np.sin(vc*t0+d)-t0*(Ey/B)
C4 = y0 - (u0/vc)*np.cos(vc*t0+d)+t0*(Ex/B)


print('x(t)=',u0/vc,'cos(',vc,'t+',d,') + ',Ey/B,'+',C3)
print('y(t)=',u0/vc,'sin(',vc,'t+',d,') - ',Ex/B,'+',C4)
print('Rx(t)=',Ey/B,'t','sin(',vc,'t+',d,') + ',C3,'-',(Ex/(B*vc)))
print('Ry(t)=',-Ex/B,'t','cos(',vc,'t+',d,') + ',C4,'-',(Ey/(B*vc)))



#εξισωσεις τροχιας
def x(t):
    return (u0/vc)*np.sin(vc*t+d)+t*(Ey/B) + C3

def y(t):
    return  (u0/vc)*np.cos(vc*t+d)-t*(Ex/B)+C4

#οδηγος κινησης
def Rx(t):
    return (Ey/B)*t + C3 - (Ex/(B*vc))

def Ry(t):
    return (-Ex/B)*t + C4 - (Ey/(B*vc))


t1 = np.linspace(0,100,50)

plt.plot(x(t1),y(t1),'b-')
plt.plot(Rx(t1),Ry(t1),'r--')
plt.legend(['x(t),y(t)','Rx(t),Ry(t)'])
plt.ylabel('y(t),Ry(t)')
plt.xlabel('x(t),Rx(t)')
plt.show()






