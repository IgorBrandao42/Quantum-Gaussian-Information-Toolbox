# -*- coding: utf-8 -*-
"""
Example of usage of the gaussian_dynamics class to calculate a conditional dynamics
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation 
# from matplotlib import cm

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
from quantum_gaussian_toolbox import *

# Parameters
omega = 2*np.pi*305e+3;                                                         # Particle natural frequency [Hz]
gamma = 2*np.pi*6.2998e+2;                                                      # Damping constant [Hz]
nbar_env = 3.1731e+07;                                                          # Environmental    occupation number
nbar_0   = 3.1731e+03;                                                          # Initial particle occupation number


# Matrix defining the open dynamics
A = np.array([[    0   ,  +omega ], [ -omega ,  -gamma ]]);                     # Drift matrix for harmonic potential with damping
D = np.diag([0, 2*gamma*(2*nbar_env+1)]);                                       # Diffusion matrix
N = np.zeros((2,1));                                                            # Mean noise vector


# Initial state
initial = gaussian_state("thermal", nbar_0);                                    # Thermal state with occupation number nbar_0
# initial.displace(100*1j - 250);                                                 # Apply a displacement operator
# nbar_0 = initial.occupation_number();                                           # Update the initial occupation number after the action of these operators


# Parameters for the measurement
N_ensemble = 1e+2                                                               # Number of iterations for the Monte Carlo method
C_int    = np.diag([np.sqrt(gamma), np.sqrt(gamma)])                            # System-bath interaction matrix. In this case: position-position and momentum-momentum interaction with coupling strength sqrt(gamma)
rho_bath = gaussian_state("thermal", 44.5)                                      # Gaussian state for the bath
s_list   = [1]                                                                  # Measurement parameter (s=1: Heterodyne ; s=0: Homodyne in x-quadrature ; s=Inf: Homodyne in p-quadrature)
phi_list = [0]                                                                  # Angle of the direction in phase space of the measurement


# Simulation
t = np.linspace(0, 1*(2*np.pi/omega), int(1e2));                                # Timestamps for simulation
simulation = gaussian_dynamics(A, D, N, initial);                               # Create instance of time evolution of gaussian state
states = simulation.conditional_dynamics(t, N_ensemble, C_int, rho_bath, s_list, phi_list); # Simulate the conditional dynamics


## Plot
x = 800*np.linspace(-1, +1, 150);                                               # Region to plot wigner function
p = 800*np.linspace(-1, +1, 150);
X, P = np.meshgrid(x, p);

#############################################################

# Plot the mean trajectory of the first moments in phase space

fig1, ax1 = plt.subplots()

R = np.zeros((2, len(t)))
for i in range(len(t)):
    R[:,i] = np.reshape(states[i].R, (2,))
plt.plot(R[0,:], R[1,:])

#############################################################

# Create animation of the wigner function (this takes a while to be made!)

# plot_args = {'rstride': 1, 'cstride': 1, 'cmap':
#              cm.bwr, 'linewidth': 0.01, 'antialiased': True, 'color': 'w',
#              'shade': True}

# fig = plt.figure()
# ax = fig.gca(projection='3d')               # to work in 3d

# W = states[0].wigner(X, P);
# plot = ax.plot_surface(X, P, W, **plot_args)
# ax.view_init(azim=-90, elev=90)

# def draw_fig(i):
#     W = states[i].wigner(X, P);
#     ax.clear()
#     plot = ax.plot_surface(X, P, W, **plot_args)
#     return plot,

# pam_ani = animation.FuncAnimation(fig, draw_fig, frames=lent(t), interval=30, blit=False)

# pam_ani.save('gaussian_dynamics_test.gif',writer='imagemagick')                    # save the animation as mp4 video file 

#############################################################

# fig, ax = plt.subplots()
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# W = states[0].wigner(X, P);
# im = plt.imshow(W, cmap=cm.coolwarm, interpolation='nearest', origin='lower', extent=[x[0],x[-1],p[0],p[-1]])

# def init():
#     W = states[0].wigner(X, P);
#     im = plt.imshow(W, cmap=cm.coolwarm, interpolation='nearest', origin='lower', extent=[x[0],x[-1],p[0],p[-1]])
#     return [im]

# def draw_fig(i):
#     W = states[i].wigner(X, P);
#     # surf = ax.plot_surface(X, P, W, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     im.set_array(W)
#     #surf = plt.imshow(W, cmap=cm.coolwarm, interpolation='nearest', origin='lower', extent=[-800,800,-800,800])
#     print(i)
#     return [im]

# anim = animation.FuncAnimation(fig, draw_fig, init_func=init, frames=30, interval=20, blit=True) # call the animator	 

# anim.save('gaussian_dynamics_test.gif',writer='imagemagick')                    # save the animation as mp4 video file 


