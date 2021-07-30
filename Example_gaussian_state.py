# -*- coding: utf-8 -*-
"""
Example of usage of the gaussian_state class
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""
import numpy as np

from quantum_gaussian_toolbox import *

# Creation of many different state
thermal = gaussian_state("thermal", 100)

squeezed = gaussian_state("squeezed", 1.2)
squeezed.displace(2 + 5j)

coherent = gaussian_state("coherent", 2+1j);
coherent.rotate(np.pi/6)

bipartite = squeezed.tensor_product([thermal])
bipartite.two_mode_squeezing(1.5)

partition = bipartite.partial_trace([1])

tripartite = thermal.tensor_product([squeezed, coherent])

bipartition = tripartite.only_modes([0,2])


# Retrieval of information from the gaussian states
lambda_thermal = thermal.symplectic_eigenvalues()

lambda_tri = tripartite.symplectic_eigenvalues()

p_tri = tripartite.purity()

p_coherent = coherent.purity()

S = thermal.von_Neumann_Entropy()

I = tripartite.mutual_information()

nbar_th = thermal.occupation_number()

nbar_3 = tripartite.occupation_number()

F_ac = coherent.fidelity(squeezed)

C_squeezed = squeezed.coherence()





