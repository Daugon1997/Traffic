#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
from numpy.random import randint
from numpy.random import uniform
from numpy.random import shuffle
import matplotlib.pyplot as plt
import math


# In[100]:


def trafficlightweg1(periode, timesteps):
    trafficlight=np.zeros(timesteps+1) #lijst met voor elke tijdstap of het stoplicht op plek intersectie-1 rood is(1) of groen(0)
    for i in range(timesteps+1):
        if (math.floor(i/periode)%2)==0:
            trafficlight[i]=1

    return(trafficlight)

def trafficlightweg2(periode, timesteps):
    trafficlight=np.zeros(timesteps+1) #lijst met voor elke tijdstap of het stoplicht op plek intersectie-1 rood is(1) of groen(0)
    for i in range(timesteps+1):
        if (math.floor(i/periode)%2)==1:
            trafficlight[i]=1

    return(trafficlight)


print(trafficlightweg1(5,10))
print(trafficlightweg2(5,10)) #de periode moet dus wel gelijk zijn anders krijg je botsing


# In[81]:


def updatenweg1(n, rho, timesteps, vmax, p, periode): #n=cells in road, rho=density (fractie), p = braking prob
    cars=int(n*rho) #number of cars
    cars=np.round(cars)
    initial = np.array([-1]*(n-cars)+[0]*cars) #alle auto's beginnen met snelheid nul en -1 is vrije ruimte
    shuffle(initial)
    
    data=[initial]
    
    intersectie = math.floor(n/2) #bepaal intersectie cell
    light=trafficlightweg1(periode, timesteps)
    
    i=1
    while i<=timesteps:
        updat=np.array([-1]*n)
        for x in range(n):   
                
            if data[-1][x]!=-1: #vind auto
                v=data[-1][x]
                
                if x==(intersectie-1): #plek van stoplicht, daar heeft de auto een andere def van d (andere regel)
                    if light[i]==1: #als stoplicht rood
                        d=1
                    if light[i]==0: 
                        if light[i-1]==1: #de eerste keer dat hij groen is, ga je nog niet rijden want dan kan er nog iemand op het kruispunt staan
                            d=1
                        if data[-1][(intersectie+1)%n]!=-1 and data[-1][(intersectie+2)%n]!=-1: #als file aan de andere kant van kruispunt, dan kun je niet het kruispunt oprijden
                            d=1
                        else:
                            d=1
                            while data[-1][(x+d)%n]==-1: #zoek de afstand tot de eerstvolgende auto
                                d=d+1
                        
                else:
                    d=1
                    while data[-1][(x+d)%n]==-1: #zoek de afstand tot de eerstvolgende auto
                        d=d+1
                    
                #nu de update regels uitvoeren
                if v<vmax: #acceleration
                    v=v+1
                if v>=d: #breaking
                    v=d-1
                if uniform(0,1)<=p: #random
                    v=max(0,v-1)
                    
                #nu verplaatsen
                updat[(x+v)%n]=v
        data.append(updat)
    
        i=i+1
    #print(data)
    return(data)

def updatenmatrixweg1(n, rho, timesteps, vmax, p, periode):
    data=updatenweg1(n, rho, timesteps, vmax, p, periode)
    matrix=np.zeros(shape=(timesteps+1,n))
    for i in range(timesteps+1):
        for j in range(n):
            if data[i][j]==-1:
                matrix[i,j]=0
            else:
                matrix[i,j]=1
    return(matrix)
    


# In[82]:


data=updatenmatrixweg1(10,0.5, 10,5,0.3,5 )
print(data)
plt.matshow(data, cmap='binary')
plt.xlabel('Distance on road')
plt.ylabel('Timesteps')
plt.title('Weg1')
plt.show()


# In[83]:


def updatenweg2(n, rho, timesteps, vmax, p, periode): #n=cells in road, rho=density (fractie), p = braking prob
    cars=int(n*rho) #number of cars
    cars=np.round(cars)
    initial = np.array([-1]*(n-cars)+[0]*cars) #alle auto's beginnen met snelheid nul en -1 is vrije ruimte
    shuffle(initial)
    
    data=[initial]
    
    intersectie = math.floor(n/2) #bepaal intersectie cell
    light=trafficlightweg2(periode, timesteps)
    
    i=1
    while i<=timesteps:
        updat=np.array([-1]*n)
        for x in range(n):
            if data[-1][x]!=-1: #vind auto
                v=data[-1][x]
                
                if x==(intersectie-1): #plek van stoplicht
                    if light[i]==1: #als stoplicht rood
                        d=1
                    if light[i]==0: 
                        if light[i-1]==1: #de eerste keer dat hij groen is, ga je nog niet rijden want dan kan er nog iemand op het kruispunt staan
                            d=1
                        if data[-1][(intersectie+1)%n]!=-1 and data[-1][(intersectie+2)%n]!=-1: #als file aan de andere kant van kruispunt, dan kun je niet het kruispunt oprijden
                            d=1
                        else:
                            d=1
                            while data[-1][(x+d)%n]==-1: #zoek de afstand tot de eerstvolgende auto
                                d=d+1
                        
                else:
                    d=1
                    while data[-1][(x+d)%n]==-1: #zoek de afstand tot de eerstvolgende auto
                        d=d+1
                    
                #nu de update regels uitvoeren
                if v<vmax: #acceleration
                    v=v+1
                if v>=d: #breaking
                    v=d-1
                if uniform(0,1)<=p: #random
                    v=max(0,v-1)
                    
                #nu verplaatsen
                updat[(x+v)%n]=v
        data.append(updat)
    
        i=i+1
    #print(data)
    return(data)

def updatenmatrixweg2(n, rho, timesteps, vmax, p, periode):
    data=updatenweg2(n, rho, timesteps, vmax, p, periode)
    matrix=np.zeros(shape=(timesteps+1,n))
    for i in range(timesteps+1):
        for j in range(n):
            if data[i][j]==-1:
                matrix[i,j]=0
            else:
                matrix[i,j]=1
    return(matrix)


# In[84]:


data=updatenmatrixweg1(10,0.3, 10,2,0.2, 5)
print(data)
plt.matshow(data, cmap='binary')
plt.xlabel('Distance on road')
plt.ylabel('Timesteps')
plt.title('Weg1')
plt.show()

data=updatenmatrixweg2(10,0.3, 10,2,0.2, 5)
#print(data)
plt.matshow(data, cmap='binary')
plt.xlabel('Distance on road')
plt.ylabel('Timesteps')
plt.title('Weg2')
plt.show()


# In[93]:


def lijstmetmatrices(n, rho, timesteps, vmax, p, periode): 
    dataweg1=updatenmatrixweg1(n, rho, timesteps, vmax, p, periode)
    dataweg2=updatenmatrixweg1(n, rho, timesteps, vmax, p, periode)
    intersectie = math.floor(n/2) #bepaal intersectie cell
    light=trafficlightweg1(periode, timesteps)
    lijst=[]
    
    for i in range(timesteps+1):
        matrix=np.ones((n,n))*[-1]
        if light[i]==1:
            matrix[intersectie,:]= dataweg1[i,:] #rij
            matrix[:,intersectie]= dataweg2[i,:]#kolom
            lijst.append(matrix)
        else:
            matrix[:,intersectie]= dataweg2[i,:]#kolom
            matrix[intersectie,:]= dataweg1[i,:] #rij
            lijst.append(matrix)

        
    return(lijst)
        
        
        


# In[95]:


a=lijstmetmatrices(11, 0.3, 10, 2, 0.2, 5)
print(a)


# In[98]:


matrix1=[[1,1],[2,2]]
matrix2=np.zeros((2,2))
print(matrix1)
print(matrix2)
matrix2[:,1]=[2,2]
print(matrix2)


# In[87]:


def averagevelocity1(n, rho, timesteps, vmax, p, periode): 
    cars=int(n*rho) #number of cars
    cars=np.round(cars)
    data=updatenweg1(n, rho, timesteps, vmax, p, periode)
    averageonestep=[]
    for i in range(timesteps-1):
        onestep=[]
        for j in range(n):
            if data[i][j]!=-1:
                onestep.append(data[i][j])
        averageonestep.append(sum(onestep)/cars)
    return(sum(averageonestep)/timesteps)


# In[75]:


def flow1(n, rho, timesteps, vmax, p, periode): 
    return(rho*averagevelocity1(n, rho, timesteps, vmax, p, periode))


# In[38]:


def averagevelocity2(n, rho, timesteps, vmax, p, periode):
    cars=int(n*rho) #number of cars
    cars=np.round(cars)
    data=updatenweg2(n, rho, timesteps, vmax, p, periode)
    averageonestep=[]
    for i in range(timesteps-1):
        onestep=[]
        for j in range(n):
            if data[i][j]!=-1:
                onestep.append(data[i][j])
        averageonestep.append(sum(onestep)/cars)
    return(sum(averageonestep)/timesteps)

def flow2(n, rho, timesteps, vmax, p, periode):  
    return(rho*averagevelocity2(n, rho, timesteps, vmax, p, periode))


# In[39]:


ro = np.arange(0.01, 1, 0.01)  
n=150
timesteps=100
vmax=2
p=0.3
periode=15

waardenv=[]
for i in ro:
    waardenv.append(averagevelocity1(n, i, timesteps, vmax, p, periode))
  
    
plt.scatter(ro, waardenv, s=15)
plt.xlabel('$rho$ (vehicles/cell)')
plt.ylabel('<V> (cells/tick)')
plt.title('weg1')
plt.show()
                   
waardenj=[]
for i in ro:
    waardenj.append(flow1(n, i, timesteps, vmax, p, periode))    
    
plt.scatter(ro, waardenj, s=15)
plt.xlabel('$rho$ (vehicles/cell)')
plt.ylabel('<J> (vehicles/tick)')
plt.title('weg1')
plt.show()

waardenv2=[]
for i in ro:
    waardenv2.append(averagevelocity2(n, i, timesteps, vmax, p, periode))
  
    
plt.scatter(ro, waardenv2, s=15)
plt.xlabel('$rho$ (vehicles/cell)')
plt.ylabel('<V> (cells/tick)')
plt.title('weg2')
plt.show()
                   
waardenj2=[]
for i in ro:
    waardenj2.append(flow2(n, i, timesteps, vmax, p, periode))    
    
plt.scatter(ro, waardenj2, s=15)
plt.xlabel('$rho$ (vehicles/cell)')
plt.ylabel('<J> (vehicles/tick)')
plt.title('weg2')
plt.show()


# In[ ]:





# In[ ]:




