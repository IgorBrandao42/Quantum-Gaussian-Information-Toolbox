# -*- coding: utf-8 -*-
"""
Example of usage of the gaussian_dynamics class
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib import cm

import numpy as np
from quantum_gaussian_toolbox import *

# Parameters
omega = 2*np.pi*305e+3;                                                         # Particle natural frequency [Hz]
gamma = 2*np.pi*6.2998e-4;                                                      # Damping constant [Hz]
nbar_env = 3.1731e+07;                                                          # Environmental    occupation number
nbar_0   = 3.1731e+03;                                                          # Initial particle occupation number


# Matrix definning the dynamics
A = np.array([[    0   ,  +omega ], [ -omega ,  -gamma ]]);                     # Drift matrix for harmonic potential with damping
D = np.diag([0, 2*gamma*(2*nbar_env+1)]);                                       # Diffusion matrix
N = np.zeros((2,1));                                                            # Mean noise vector


# Initial state
initial = gaussian_state("thermal", nbar_0);                                    # Thermal state with occupation number nbar_0
initial.displace(100*1j - 250);                                                 # Apply a displacement operator
# initial.squeeze(1.1);                                                         # Apply a squeezing    operator
# initial.rotate(-pi/4);                                                        # Apply a rotation     operator
nbar_0 = initial.occupation_number();                                           # Update the initial occupation number after the action of these operators


# Simulation
t = np.linspace(0, 5*2*np.pi/omega, int(1e3));                                  # Timestamps for simulation
simulation = gaussian_dynamics(A, D, N, initial);                               # Create instance of time evolution of gaussian state
states = simulation.run(t);                                                     # Simulate


## Plot
x = 800*np.linspace(-1, +1, 150);                                               # Region to plot wigner function
p = 800*np.linspace(-1, +1, 150);
X, P = np.meshgrid(x, p);

fig, ax = plt.subplots()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

W = states[0].wigner(X, P);
im = plt.imshow(W, cmap=cm.coolwarm, interpolation='nearest', origin='lower', extent=[-800,800,-800,800])
#fig.colorbar(im, shrink=0.5)

# def draw_fig(i):
#     W = states[i].wigner(X, P);
#     # surf = ax.plot_surface(X, P, W, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     im.set_array(W)
#     #surf = plt.imshow(W, cmap=cm.coolwarm, interpolation='nearest', origin='lower', extent=[-800,800,-800,800])
#     print(i)
#     return [im]

# # call the animator	 
# anim = animation.FuncAnimation(fig, draw_fig, frames=100, interval=20, blit=True) 

# # save the animation as mp4 video file 
# anim.save('gaussian_dynamics_test.gif',writer='imagemagick')


