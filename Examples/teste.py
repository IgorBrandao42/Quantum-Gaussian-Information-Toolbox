# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:24:57 2021

@author: igorb
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gaussian_state import gaussian_state

a = gaussian_state("thermal", 100)

b = gaussian_state("squeezed", 1.2)
b.displace(2 + 5j)

b.tensor_product([b])
bb = b.tensor_product([b])
bb.two_mode_squeezing(0.5)

b = bb.partial_trace([1])

c = gaussian_state("coherent", 2+1j);

tripartite = a.tensor_product([b,c])

single = tripartite.partial_trace([0,2])

bipartite = tripartite.partial_trace([2])

single2 = tripartite.only_modes([0,2])

lambda_a = a.symplectic_eigenvalues()

lambda_tri = tripartite.symplectic_eigenvalues()

tripartite.purity()

c.purity()

S = c.von_Neumann_Entropy()

I = tripartite.mutual_information()

#nbar_th = c.occupation_number()

nbar3 = tripartite.occupation_number()

a.fidelity(c)

x = np.linspace(-10, 10, 200)
X, P = np.meshgrid(x,x)
W = c.wigner(X,P)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, P, W, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5)
