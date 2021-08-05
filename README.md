
# Gaussian Quantum Information Numerical Toolbox (Python)

This is a object-oriented Python library aimed at simulating any multimode quantum gaussian states, findiing their time evolution according to sets of quantum Langevin and Lyapunov equations and recovering the information about these states.

## gaussian_state class
Gaussian states are a particular class of continuous variable states that can be completelly described by their quadratures' first and second moments [[Rev. Mod. Phys. 84, 621]](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621).  The toolbox is able to simulate any gaussian state through the class 'gaussian_state' whose input can be

 - R, V --- mean quadrature vector and covariance matrix for the desired gaussian states;
- Name-pair value for an elementary single-mode gaussian state:
	 - "vaccum" --- generates a vaccum state
	 - "coherent", 1+2j --- generates a coherent state with complex amplitude 1+2j
	 - "squeezed", 2 --- generates a squeezed state with real squeezing parameter 2
	 - "thermal", 0.1 --- generates a thermal state with occupation number 0.1

```python
import numpy as np
from quantum_gaussian_toolbox import *

vacuum_state   = gaussian_state("vacuum");           # Vacuum   state
coherent_state = gaussian_state("coherent", 1 + 2j); # Coherent state
squeezed_state = gaussian_state("squeezed", 2);      # Squeezed state
thermal_state  = gaussian_state("thermal", 0.1);     # Thermal  state

R = np.array([1, 2, 3, 4])                           # Mean quadrature vector
V = np.array([[1, 0, 0, 0],                          # Covariance matrix
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
              
bipartite_state = gaussian_state(R, V);				 # Generic multimode gaussian state
```

More so, it is possible to combine any number of gaussian states into their direct product and retrieve only partitions of the composite gaussian state:
```python
tripartite_state = thermal.tensor_product([vaccum, generic_state]); # Create tripartite gaussian state

bipartition = tripartite_state.partial_trace([2]);    # Perform partial trace over the third mode

last_two_modes = tripartite_state.only_modes([1, 2]); # Get the last two modes by performing partial trace over the first mode 
```

It is possible to apply a number of gaussian unitaries to a given single-mode or bipartite state, namelly:
 1. Single-mode operators:
	 - Displacement operator
	 - Squeezing operator
	 - Rotation operator
 2. Two-mode operators:
	 - Beam-splitter operator
	 - Two-mode squeezing operator

```python
a = squeezed.displace(3 + 4j); # Apply displacement operator

b = thermal.squeeze(2);        # Apply squeezing operator

c = a.rotate(pi/2);		       # Apply rotation operator

d = bipartite_state.beam_splitter(1.5) # Apply beam splitter operator

e = bipartite_state.two_mode_squeezing(2) # Apply two-mode squeezing operator
```

Information about a generic multimode gaussian state can be recovered through the other 'gaussian_state' class' methods:

| Method | Calculates|
|--|--|
| purity | Purity |
| symplectic_eigenvalues | Symplectic eigenvalues of the covariance matrix |
| von_Neumann_Entropy | von Neumann entropy |
| mutual_information | Mutual Information for multimode state |
| occupation_number | Array with ccupation number for each mode of the gaussian state |
| wigner | Wigner function over a 2D grid for a single mode gaussian state |
|squeezing_degree | Ratio of the variance of the squeezed and antisqueezed quadratures|
|fidelity | Quantum Fidelity between the two gaussian states|
| logarithmic_negativity | Logarithmic negativity for a bipartition of a gaussian state|

Please refer to the [initial  documentation pdf](https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python/blob/main/Documentation%20-%20Quantum_Gaussian_Information_Toolbox%20-%20python.pdf) for a more indepth description of these methods

## gaussian_dynamics class
The toolbox is also equipped with a second class 'gaussian_dynamics' to simulate the time evolution of a given initial state (gaussian_state) following a gaussian preserving dynamics dictated by an arbitrary set of quantum Langevin and Lyapunov equations

The input arguments to this class constructor are:

 - A --- Drift matrix for the Langevin equation (numpy array or function)
 - D --- Diffusion matrix for the Lyapunov equation (numpy array)
 - N --- Vector with the mean value of the input noises (numpy array)
 - initial_state --- Initial state for the time evolution (gaussian_state class instace)

The toolbox is able to account for time dependent drift matrices given by gufunc or lambda functions!

See below a simple example:
```python
import numpy as np
from quantum_gaussian_toolbox import  *

omega = 2*pi*197e+3                            # Particle natural frequency [Hz]
gamma = 2*pi*881.9730                          # Damping constant [Hz] at 1.4 mbar pressure
nbar_env = 3.1731e+07                          # Environmental    occupation number

A = np.block([[    0   ,  +omega ]             # Drift matrix for harmonic potential
              [ -omega ,  -gamma ]]) 
        
D = np.diag([0, 2*gamma*(2*nbar_env+1)])       # Diffusion matrix
N = np.zeros((2,1))                            # Mean noise vector

alpha = 1 + 2j                                 # Coherent state amplitude
initial_state = gaussian_state("coherent",alpha) # Initial state

t = linspace(0, 2*pi/omega, 1000);              # Timestamps for simulation
simulation = gaussian_dynamics(A, D, N, initial_state); # Create simulation instance!
states = simulation.run(t);                    # Simulate and retrieve time evolved states (array of gaussian_state instances)   
```

The method 'run' returns the time evolved state (array of gaussian_state instances) at the specified times (input argument)


## Dependencies

This toolbox makes use of the numpy and scipy packages.

## Installation

Clone this repository or download **quantum_gaussian_toolbox.py** file to your project folder and import the toolbox:

```python
from quantum_gaussian_toolbox import *
```

# Running Example
In the file **Example_gaussian_state.py** there is a basic example of the capabilities of this Toolbox to simulate a multimode gaussian state and retrieve information from it.

In the file **Example_gaussian_dynamics.py** there is a basic example of the capabilities of this Toolbox to simulate the time evolution of multimode gaussian state following closed/open quantum dynamics through a set of quantum Langevin and Lyapunov equations.

## Author
[Igor Brandão](mailto:igorbrandao@aluno.puc-rio.br) - M.Sc. in Physics from Pontifical Catholic University of Rio de Janeiro, Brazil. Advisor: [Thiago Guerreiro](mailto:barbosa@puc-rio.br)

## Mathematical Formalism
For the study of Gaussian Quantum Information, this code was based on and uses the same formalism as:

> Christian Weedbrook, Stefano Pirandola, Raúl García-Patrón, Nicolas J. Cerf, Timothy C. Ralph, Jeffrey H. , "Gaussian quantum information", [Rev. Mod. Phys. 84, 621](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621)

For the quantum fidelity, see:
> L. Banchi, S. L. Braunstein, S. Pirandola, " Quantum Fidelity for Arbitrary Gaussian States", [[Phys. Rev. Lett. 115, 260501]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.260501)

For the coherence, see:
> J. Xu, "Quantifying coherence of Gaussian states", [[Phys. Rev. A 93, 032111 (2016)]](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032111).

## License
This code is made available under the Creative Commons Attribution - Non Commercial 4.0 License. For full details see LICENSE.md.

Cite this toolbox as: 
> Igor Brandão, "Quantum Gaussian Information Numerical Toolbox", https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python. Retrieved <*date-you-downloaded*>


## Acknowledgment
The author thanks Daniel Ribas Tandeitnik and Professor Thiago Guerreiro for the discussions. The author is thankful for support received from FAPERJ Scholarship No. E-26/200.270/2020 and CNPq Scholarship No. 140279/2021-0.



