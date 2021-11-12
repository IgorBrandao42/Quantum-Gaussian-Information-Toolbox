

# QuGIT: Quantum Gaussian Information Toolbox

QuGIT is an open-sourced object-oriented Python library aimed at simulating multimode quantum gaussian states, finding their unconditional or conditional time evolution according to sets of quantum Langevin and Lyapunov equations and recovering information about these states.

Gaussian states are a particular class of continuous variable states that can be completelly described by their quadratures' first and second moments [[Rev. Mod. Phys. 84, 621]](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621).  

The toolbox takes advantage of the structure of gaussian quantum states to efficiently simulate them without the need to work with truncated Hilbert spaces.


## gaussian_state class

The fundamental building block of the toolbox is its ability to emulate an arbitrary multimode gaussian state, perform gaussian operations, and retrieve information from it. This is achieved through the custom Python class `gaussian_state`, whose constructor arguments are
 - R, V --- numpy.ndarray mean quadrature vector and covariance matrix for the desired gaussian states;

Elementary gaussian states can be created using the library functions: `vaccum()`, `coherent()`, `squeezed()` and `thermal()`.

```python
import numpy as np
import quantum_gaussian_toolbox as qgt

vac = qgt.vacuum()           				# Vacuum   state
coh = qgt.coherent(1-20j)       			# Coherent state
sq  = qgt.squeezed(1.2)      				# Squeezed state
th  = qgt.thermal(4)             			# Thermal  state

R = np.array([1, 2, 3, 4])                  # Mean quadrature vector
V = np.eye(4)                 				# Covariance matrix
state0 = qgt.gaussian_state(R, V)			# Multimode gaussian state
```

It is possible to apply a number of gaussian operations and retrieve information about the resulting state:

```python
### Tensor product and partial trace
state0.tensor_product([vaccum, th])      		# Update state0 to be the tensor product of itself and the state on the argument

tripartite = qgt.partial_trace(state0,[2])  	# Tripartite is a copy of state0 after partial trace was performed on its 3rd mode. state0 is unchanged

bipartite = tripartite_state.only_modes([1, 2]) # Get the last two modes by performing partial trace over the first and second modes


### Gaussian unitaries
sq.displace(3 + 4j); 	          # Apply displacement operator

th.squeeze(2);        	          # Apply squeezing operator

bipartite.two_mode_squeezing(2)   # Apply two-mode squeezing operator


### General-dyne measurements
bipartite.measurement_heterodyne(coh) # Updates the global state 
# after the last mode was measured into a coherent state


### Retrieve information
p = th.purity()					  # Calculates the purity of a state

sq_degree = squeezing_degree(sq)  # Calculates the amount of squeezing
```

Please refer to the [initial  documentation pdf](https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python/blob/main/Documentation%20-%20Quantum_Gaussian_Information_Toolbox%20-%20python.pdf) for a more indepth description of these and other methods!

## gaussian_dynamics class
The toolbox is also equipped with a second class 'gaussian_dynamics' to simulate unconditional and conditional time evolution of a given initial state (`gaussian_state`) following a gaussian preserving dynamics dictated by an arbitrary set of quantum Langevin and Lyapunov equations and general-dyne measurements.

Example of usage:
```python
import numpy as np
import quantum_gaussian_toolbox as qgt

omega = 2*np.pi*197e+3                         # Particle natural frequency [Hz]
gamma = 2*np.pi*881.9730                       # Damping constant [Hz] at 1.4 mbar pressure
nbar_env = 3.1731e+07                          # Environmental    occupation number

A = np.block([[    0   ,  +omega ],            # Drift matrix for harmonic potential
              [ -omega ,  -gamma ]]) 
        
D = np.diag([0, 2*gamma*(2*nbar_env+1)])       # Diffusion matrix
N = np.zeros((2,1))                            # Driving vector

alpha = 1 + 2j                                 # Coherent state amplitude
initial_state = qgt.coherent( alpha) 		   # Initial state

t = np.linspace(0, 2*np.pi/omega, 1000);       # Timestamps for simulation

simulation = qgt.gaussian_dynamics(A, D, N, initial_state) # Simulation instance

states = simulation.unconditional_dynamics(t)  # Simulate
# Retrieve a list of time evolved gaussian_state class instances
```

We note that  the toolbox is able to account for time dependent drift matrices given by gufunc or lambda functions. Please refer to the [initial  documentation pdf](https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python/blob/main/Documentation%20-%20Quantum_Gaussian_Information_Toolbox%20-%20python.pdf) for a more indepth description of these methods


## Dependencies

The toolbox makes use of the Numpy and Scipy packages.

## Installation

Clone this repository or download **quantum_gaussian_toolbox.py** file to your project folder and import the toolbox:

```python
import quantum_gaussian_toolbox as qgt
```

# Running Example
In the file **Example_gaussian_state.py** there is a basic example of the capabilities of this Toolbox to simulate a multimode gaussian state and retrieve information from it.

In the file **Example_gaussian_dynamics.py** there is a basic example of the capabilities of this Toolbox to simulate the time evolution of multimode gaussian state following closed/open quantum dynamics through a set of quantum Langevin and Lyapunov equations.

## Author
 Igor Brandão, M. Sc. in Physics from Pontifical Catholic University of Rio de Janeiro, Brazil.
 [Contact me](mailto:igorbrandao@aluno.puc-rio.br) --- [ Google Scholar](https://scholar.google.com.br/citations?user=WuywvSEAAAAJ) --- [Research Gate](https://www.researchgate.net/profile/Igor-Brandao-2)

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
> Igor Brandão, "Quantum Gaussian Information Toolbox", https://github.com/IgorBrandao42/Quantum-Gaussian-Information-Toolbox. Retrieved <*date-you-downloaded*>


## Acknowledgment
The author thanks Daniel Ribas Tandeitnik and Professor Thiago Guerreiro for helpful discussions, and Professor Dan Marchesin for all the coding lessons. The author is thankful for support received from FAPERJ Scholarships No. E-26/200.270/2020 and CNPq Scholarship No. 140279/2021-0.



