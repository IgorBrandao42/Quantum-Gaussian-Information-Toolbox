# Gaussian Quantum Information Numerical Toolbox (Python)

This is a Python Toolbox for numerical simulation of quantum gaussian states and their dynamics dictated by a set of linear Langevin and Lyapunov equations

For the full description and notation used for gaussian quantum states, please refere to [[Rev. Mod. Phys. 84, 621]](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621). Particularly, for the quantum fidelity, see [[Phys. Rev. Lett. 115, 260501]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.260501); for the coherence, see [[Phys. Rev. A 93, 032111 (2016)]](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032111).

The README below is not complete, but for the moment you can refer to the [documentation pdf](https://github.com/IgorBrandao42/Quantum-Gaussian-Information-Toolbox/blob/main/Documentation__Quantum_Gaussian_Information_Toolbox.pdf)!

## Installation

Clone this repository or download this Toolbox to your project folder and import the toolbox:

```python
from quantum_gaussian_toolbox import *
```

## Usage
 Creation of gaussian states

```MATLAB
import numpy as np
from gaussian_state import gaussian_state
from gaussian_dynamics import gaussian_dynamics

nbar_0   = 1.0577e+05;                             # Initial particle occupation number
thermal_state = gaussian_state("thermal", nbar_0); # Initial state
thermal_state.squeeze(3);                          # Apply squeezing operator to gaussian state

R = np.array([10, 5])                              # Mean quadrature vector
V = np.array([[1, 0],                              # Covariance matrix
              [0, 1]])
              
generic_state = gaussian_state(R, V);

bipartite_state = thermal.tensor_product(generic_state); # Create bipartite gaussian state
```

Calculation of time evolution:
```python
import numpy as np
from gaussian_state import gaussian_state
from gaussian_dynamics import gaussian_dynamics

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

#### Running Example
In the file **Example_gaussian_state.py** there is a basic example of the capabilities of this Toolbox to simulate a multimode gaussian state and retrieve information from it.

In the file **Example_gaussian_dynamics.py** there is a basic example of the capabilities of this Toolbox to simulate the time evolution of multimode gaussian state following closed/open quantum dynamics through a set of quantum Langevin and Lyapunov equations.

## Author
[Igor Brandão](mailto:igorbrandao@aluno.puc-rio.br) - M.Sc. in Physics from Pontifical Catholic University of Rio de Janeiro, Brazil. Advisor: [Thiago Guerreiro](mailto:barbosa@puc-rio.br)

## Mathematical Formalism
For the study of Gaussian Quantum Information, this code was based on and uses the same formalism as:

> Christian Weedbrook, Stefano Pirandola, Raúl García-Patrón, Nicolas J. Cerf, Timothy C. Ralph, Jeffrey H. , "Gaussian quantum information", [Rev. Mod. Phys. 84, 621](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621)

## License
This code is made available under the Creative Commons Attribution - Non Commercial 4.0 License. For full details see LICENSE.md.

Cite this toolbox as: 
> Igor Brandão, "Quantum Gaussian Information Numerical Toolbox", https://github.com/IgorBrandao42/Quantum-Gaussian-Information-Numerical-Toolbox. Retrieved <*date-you-downloaded*>


## Acknowledgment
The author thanks Daniel Ribas Tandeitnik and Professor Thiago Guerreiro for the discussions. The author is thankful for support received from FAPERJ Scholarship No. E-26/200.270/2020 and CNPq Scholarship No. 140279/2021-0.



