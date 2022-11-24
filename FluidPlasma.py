# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:05:07 2019

@author: AM256188
"""

from numba import jit #library for calculation in C
import numpy as np
import sys
import math
import cmath
import os
import glob
import h5py as h5
import copy
import time
import random as rand
from scipy.interpolate import griddata
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy.signal import chirp, sweep_poly
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib import animation
import matplotlib.cm as cm
import time
import h5py as h5
import random      



def set_bnd(b, x, N):
    for i in range(N):
        if b==1:
#            x[0,i]=-x[1,i]
#            x[N,i]=-x[N-1,i]
            x[0,i]=-x[N,i]
            x[N,i]=-x[N-1,i]
        else:
            x[0,i]=x[1,i]
            x[N,i]=x[N-1,i]
        if b==2:
            x[i,0]=x[N,1]
            x[i,N]=x[i,N-1]
        else:
            x[i,0]=x[i,N]
            x[i,N]=x[i,N-1]
    x[0,0]=0.5*(x[1,0]+x[0,1])
    x[0,N]=0.5*(x[1,N]+x[0,N-1])
    x[N,0]=0.5*(x[N-1,0]+x[N,1])
    x[N,N]=0.5*(x[N-1,N]+x[N,N-1])
    return x

def lin_solve(b, x, x0, a, co, N):
    x=np.array(x)
    x0=np.array(x0)
    for n in range(10):
        for i in range(1,N):
            for j in range(1,N):
                x[i,j] = ( x0[i,j]+a* ( x[i-1,j] + x[i+1,j] + x[i,j-1] + x0[i,j+1] ) )/co
        x=set_bnd(b, x, N)
    return x

def diffuse (b, x, x0, diff, dt, N):
    a = dt * diff * N * N
    x=lin_solve(b, x, x0, a, 1 + 4 * a, N)
    return x

def advect(b, d, d0,  velocX, velocY, dt, N):
    dtx = dt * (N-2)
    dty = dt * (N-2)
    for i in range(1,N):
        for j in range(1,N):
            x=i-dtx*velocX[i,j]
            y=j-dty*velocY[i,j]                
            if x < 0.5:
                x = 0.5
            if x > N + 0.5:
                x = N + 0.5
            i0 = math.floor(x)
            i1 = i0 + 1
            if y < 0.5:
                y = 0.5
            if y > N + 0.5:
                y = N + 0.5 
            j0 = math.floor(y)
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            d[i, j] = s0 * ( t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    d=set_bnd(b, d, N)
    return d

def project(velocX, velocY, p, div, N):
    for i in range(1,N):
        for j in range(1,N):
            div[i, j] = -0.5*(velocX[i+1, j] - velocX[i-1, j] +velocY[i  , j+1]-velocY[i  , j-1])/N
            p[i, j] = 0
    div=set_bnd(0, div, N)
    p=set_bnd(0, p, N)
    p=lin_solve(0, p, div, 1, 6, N)

    for i in range(1,N):
        for j in range(1,N):
            velocX[i, j] = velocX[i, j]-0.5* (  p[i+1, j]-p[i-1, j] ) * N
            velocY[i, j] = velocY[i, j]-0.5* (  p[i, j+1]-p[i, j-1] ) * N
    velocX=set_bnd(1, velocX, N)
    velocY=set_bnd(2, velocY, N)
    return velocX, velocY
class FluidCube:
    def __init__(self, size, diffusion, viscosity, dt):
        self.N = size
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        self.s = np.full((self.N+1,self.N+1), 0.)
        self.density = np.full((self.N+1,self.N+1), 0.)
        self.Vx = np.full((self.N+1,self.N+1), 0.)
        self.Vy = np.full((self.N+1,self.N+1), 0.)
        self.Vx0 = np.full((self.N+1,self.N+1), 0.)
        self.Vy0 = np.full((self.N+1,self.N+1), 0.)
        
    def clear(self):
        self.s = []
        self.density = []
        self.Vx = np.full((self.N+1,self.N+1), 0.)
        self.Vy = np.full((self.N+1,self.N+1), 0.)
        self.Vx0 = np.full((self.N+1,self.N+1), 0.)
        self.Vy0 = np.full((self.N+1,self.N+1), 0.)

    def FluidCubeAddDensity(self, amount):
        self.density=self.density + amount
        
    def FluidCubeAddVelocity(self, amountX, amountY):
        self.Vx =self.Vx + amountX
        self.Vy =self.Vy + amountY
     
    def FluidCubeStep(self):
        A0=self.density
        self.s=self.density
        self.density=A0
        self.density = diffuse(0, self.s, self.density, self.diff, self.dt, self.N)
        A0=self.density
        self.s=self.density
        self.density=A0       
        self.density = advect(0, self.s, self.density, self.Vx, self.Vy, self.dt, self.N)
        
        Ax=self.Vx0
        Ay=self.Vy0
        self.Vx0=self.Vx
        self.Vy0=self.Vy
        self.Vx=Ax
        self.Vy=Ay
        
        self.Vx = diffuse(1, self.Vx, self.Vx0, self.visc, self.dt, self.N)
        self.Vy = diffuse(2, self.Vy, self.Vy0, self.visc, self.dt, self.N)
    
        Ax=self.Vx0
        Ay=self.Vy0
        self.Vx0=self.Vx
        self.Vy0=self.Vy
        self.Vx=Ax
        self.Vy=Ay
        
        self.Vx = advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt, self.N)
        self.Vy = advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt, self.N)
    
        self.Vx, self.Vy = project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.N)
    
#simulation parameters  
size=300 #grid size
sim_t=64 #time points
back_flow=0.0 #additional background flow in y direction otherwise everything moves 1 point per time point
dt=0.01 
#physical parameters
diffusion=0.00001/size/size
viscosity=1
dv_max=0 #velocity fluctuation
dv_shear=0.16 #max velcoity of the parabolic sheared flow
fluc=0 #random small fluctuations yes or no 1 or 0
n_max=1 #density max of the linear profile
mode_A=0.1 #mode amplitude of the density
#geometry
T=int(size/2) #mode period size/2 gives 4 periods of the mode in y direction
shift_pol=0 #to shift mode in the y direction
pos_0=250 #position of the 0 density of the linear profile 
pos_shear=250 #position of the shear center
shear_half_width=50 
L=2*shear_half_width # modes have two rows within the sheared flow 
tilt=0 #tilt 0 or 1 for 45 degrees tilt

#grid initialization
vxxx=np.full((size+1,size+1),0.000)
vyyy=np.full((size+1,size+1),0.000)
vyyy0=np.full((size+1,size+1), 0.000)
vxxx0=np.full((size+1,size+1),0.000)
NE=np.full((size+1,size+1),0.0) 

for ii in range(size+1):
    for jj in range(size+1):
        vxxx[ii,jj]=dv_max*math.sin(ii/L+fluc*random.random())*math.sin(jj/T+fluc*random.random())
for ii in range(size+1):
    for jj in range(size+1):
        vyyy[ii,jj]=dv_max*math.cos(ii/L+fluc*random.random())*math.cos(jj/T+fluc*random.random())
for ii in range(pos_shear-shear_half_width,pos_shear+shear_half_width):
    for jj in range(size+1):
        vyyy0[ii,jj]=vyyy0[ii,jj]+dv_shear*(1-(ii-pos_shear)**2/shear_half_width**2)     

for jj in range(size+1):
    NE[0:pos_0,jj]=np.linspace(n_max,0,pos_0)
    for jjj in range(pos_0-10,pos_0+10):
           NE[jjj-1,jj]=n_max*(jjj-pos_0-10)**2/(pos_0)/40
   
      
for ii in range(2*shear_half_width-1):
    for jj in range(size+1):
        NE[ii+pos_shear-shear_half_width,jj]= NE[ii+pos_shear-shear_half_width,jj]+mode_A*(math.sin(2*math.pi*(ii)/L+fluc*random.random())*math.sin(2*math.pi*(jj+2*tilt*abs(ii-shear_half_width)/2+shift_pol)/T+fluc*random.random()))**2
plt.pcolor(NE)
     
      

simul0 = FluidCube(size, diffusion, viscosity, dt)
simul0.clear()
simul0.N=size
simul0.density=np.abs(NE)
simul0.s=np.abs(NE)  

simul0.Vx0=vxxx 
simul0.Vy0=vyyy
simul0.Vx=vxxx0
simul0.Vy=vyyy0
frame=[]
for i in range(sim_t):

    frame.append(simul0.density)
    simul0.FluidCubeStep()
    simul0.Vx0=simul0.Vx
    simul0.Vy0=simul0.Vy
    a1=simul0.density[:,size]
    a2=simul0.s[:,size]
    a3=simul0.Vx[:,size]
    a4=simul0.Vy[:,size]
    a5=simul0.Vx0[:,size]
    a6=simul0.Vy0[:,size]   
    for jj in range(size,0,-1):
        simul0.density[:,jj]=simul0.density[:,jj-1]
        simul0.s[:,jj]=simul0.s[:,jj-1]
    simul0.density[:,0]=a1
    simul0.s[:,0]=a2

frame=np.array(frame)

fig, ax = plt.subplots()
image=np.array(np.mean(frame,0)+10*(frame[0,:,:]-np.mean(frame,0)))
cax = ax.imshow(np.array(image), vmin=0.0, vmax=1.1, extent=[0, size+1, 0, size+1], aspect='auto', cmap=cm.seismic)
ax.set_title('shear flow max $v_y$='+str(dv_shear))

   
def animate(i):
     cax.set_array(np.mean(frame,0)+10*(np.array(frame[i,:, :])-np.mean(frame,0)))

anim = animation.FuncAnimation(fig, animate, interval=30, frames=sim_t)
anim.save('Turbulence_map.gif')
plt.show()

hf = h5.File('Turbulence_map.h5', 'w')
hf.create_dataset('ne_map', data=frame)
hf.close()
