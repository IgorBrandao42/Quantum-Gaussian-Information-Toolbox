[![View Gaussian Quantum State Toolbox on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/87614-quantum-open-dynamics-and-gaussian-information-toolbox)

# Gaussian Quantum State Toolbox

This is a MATLAB Toolbox for numerical simulation of quantum gaussian states and their dynamics dictated by a set of linear Langevin and Lyapunov equations


In regards to the numerical Gaussian Quantum Information portion of this toolbox, given a gaussian state's expected quadratures and/or covariance matrix, it calculates:
Its wigner function, von Neumann entropy, logarithmic negaitivy, mutual information, its covariance matrix's symplectic_eigenvalues and single mode partitions and bipartitions. Given two arbitrary gaussian states, it also computes their quantum fidelity. For the full description and notation used for gaussian quantum states, please refere to [[Rev. Mod. Phys. 84, 621]](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621). For the quantum fidelity, see [[Phys. Rev. Lett. 115, 260501]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.260501).

## Installation

Clone this repository or download this Toolbox and add its main folder to the MATLAB path:

```MATLAB
addpath('<download-path>/<name-folder>');
```

## Usage

This program only inputs are the parameters values and time interval for the calculation. The program infers the number of particles from the lengths of these vectors.
```MATLAB
% Define the constants for the system under study (in this case 3 particles and one cavity)
omega =    2*pi*[305.4e+3;  305.4e+3;  305.4e+3];     % Natural frequency of the particles [Hz]
g     =    2*pi*[ 64.0e+3;   93.2e+3;  109.2e+3];     % Coupling strength                  [Hz]
gamma =    2*pi*[ 9.57e-4;   9.57e-4;   9.57e-4];     % Damping                            [Hz]
T     =         [ 4.6e-6;    4.6e-6;     4.6e-6];     % Initial temperature of each particle             [K]
T_environment = [   300 ;       300;        300];     % Temperature for the environment of each particle [K]

Delta = 2*pi*315e+3;                                  % Cavity-tweezer detuning            [Hz] (frequency of the optical mode)
kappa = 2*pi*193e+3;                                  % Cavity linewidth                   [Hz]

t = linspace(0, 4.2e-6, 1e+3);                        % Time interval for the simulation   [s]
```

You need to create an instance of a simulation (class) with the parameters and run the calculations at the time stamps:
```MATLAB
% Create an instance of a simulation
example = simulation(omega, g, gamma, T, T_environment, Delta, kappa);

% Run the every calculation available
example.run(t);
```

The simulations are a handle class and its results can be directly extracted or plotted using its internal plotting methods:
```MATLAB
% Plot the results
example.plot();
```

The user can choose to calculate only what suits them, by passing extra parameters to the method 'run', they are:

|  Optional parameter  | Specific calculation of time evolution |
|----------------------|----------------------|
| "langevin"           | Solve semiclassical Langevin equations for the expectation value of the quadratures |
| "lyapunov"           | Solve Lyapunov equation for the covariance matrix| 
| "steady_state"       | Find steady state covariance matrix |
|"occupation_number"   | Find the occupation number for each mode|
| "entanglement"       | Calculate the logarithmic negativity for each bipartition |
| "entropy"            | Calculate the von Neumann entropy for each mode, bipartition and the whole system|
| "mutual_information" | Calculate the mutual information for the whole system|
| "fidelity_test"      | Approximate each mode state by a thermal state through Fideliy, finding the effective temperature |

```MATLAB
% Run a specific calculation
example.run(t, "occupation_number");
```

#### Running Example
In the file **Example_simulation.m** there is a basic example of the capabilities of this Toolbox.

## Author
[Igor Brandão](mailto:igorbrandao@aluno.puc-rio.br) - Master's student in [Thiago Guerreiro](mailto:barbosa@puc-rio.br)'s Lab at Pontifical Catholic University of Rio de Janeiro, Brazil

## Mathematical Formalism
For the optomechanics time evolution, this codes was originally created for and uses the same formalism as:
> Igor Brandão, Daniel Tandeitnik, Thiago Guerreiro, "Coherent Scattering-mediated correlations between levitated nanospheres", [arxiv:2102.08969](https://arxiv.org/abs/2102.08969) 

For the study of Gaussian Quantum Information, this code was based on and uses the same formalism as:

> Christian Weedbrook, Stefano Pirandola, Raúl García-Patrón, Nicolas J. Cerf, Timothy C. Ralph, Jeffrey H. , "Gaussian quantum information", [Rev. Mod. Phys. 84, 621](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621)

## License
This code is made available under the Creative Commons Attribution - Non Commercial 4.0 License. For full details see LICENSE.md.

Cite this toolbox as: 
> Igor Brandão, "Quantum Open Dynamics and Gaussian Information : Linear Optomechanics", [https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Toolbox-for-Linear-Optomechanics](https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Toolbox-for-Linear-Optomechanics). Retrieved <*date-you-downloaded*>

## File Listing

|          File name          |                               What it does                                                            |
|-------------------|-------------------------------------------------------------------------------------------------------|
|        simulation.m         |     Class definning a simulation of N particles interacting with an optical cavity                    |
|         particle.m          |                    Class definning a nanoparticle                                                     |
|      optical_cavity.m       |                   Class definning an optical cavity                                                   |
|-------------------|------------------------------------------------------------------------------------|
|  symplectic_eigenvalues.m   |            Calculates the sympletic eigenvalues of a covariance matrix                                |
|          wigner.m           | Calculates the wigner function for a gaussian state from its mean value of the quadrature and its covariance matrix |
|         fidelity.m          |          Calculates the fidelity between the two arbitrary gaussian states from its mean value of the quadrature and its covariance matrix                         |
|    mutual_information.m     | Calculates the mutual information of a multipartite gaussian state from its covariance matrix
|   von_Neumann_Entropy.m     |  Calculates the von Neumann entropy     of a multipartite gaussian state from its covariance matrix   |
|  logarithmic_negativity2.m  |   Calculates the logarithmic negativity of a bipartite   gasussian state from its covariance matrix   |
|      single_mode_CM.m       |     Finds the covariance submatrix for a single mode from the full covariance matrix (partial trace)  |
|       bipartite_CM.m        |     Finds the covariance submatrix for a bipartition from the full covariance matrix (partial trace)  |
|-------------------|------------------------------------------------------------------------------------|
|        lyapunov_ode         |       ODE that defines the Lyapunov equation to be integrated by ode45                                |
|          func.m             |         Auxiliar mathematical function for the von Neumann entropy                                    |
|         Example.m           |                 Basic example of usage of the Toolbox                                                 |

## Acknowledgment
The author thanks Daniel Ribas Tandeitnik and Professor Thiago Guerreiro for the discussions. The author is thankful for support received from FAPERJ Scholarship No. E-26/200.270/2020



