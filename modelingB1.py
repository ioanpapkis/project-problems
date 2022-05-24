# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:03:29 2020

@author: johnp
"""

import numpy as np
import matplotlib.pyplot as p


t0 = float(input('t0:'))
x0 = float(input('x(t0):'))
y0 = float(input('y(t0):'))


# Σταθερες της εξίσωσης
d1 = np.exp(2*t0)*((y0-3)*np.sin(3*t0)+(x0-4)*np.cos(3*t0))
d2 = (np.exp(2*t0)/np.cos(3*t0))*((y0-3)-((y0-3)*np.sin(3*t0)-(x0-4)*np.cos(3*t0))*np.sin(3*t0))




def x(t):
    return np.exp(-2*t)*(d1*np.cos(3*t)-d2*np.sin(3*t)) + 4


def y(t):
    return np.exp(-2*t)*(d1*np.sin(3*t)+d2*np.cos(3*t)) + 3



    


t1 = np.linspace(0,10,1000)



p.plot(x(t1),y(t1),'b-')
p.legend(['linear'])
p.xlabel('x')
p.ylabel('y')
p.grid()
p.show()




