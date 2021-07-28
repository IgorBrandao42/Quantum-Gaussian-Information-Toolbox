# -*- coding: utf-8 -*-
"""
Class simulating the time evolution of a gaussian state
Github: https://github.com/IgorBrandao42/Gaussian-Quantum-Information-Numerical-Toolbox-python

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
import types

def is_lambda_function(obj):
    """
    Auxiliar internal function checking if a given variable is a lambda function
    """
    return isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"

def lyapunov_ode(t, V_old_vector, A, D):
    """
    Auxiliar internal function defining the Lyapunov equation 
    and calculating the derivative of the covariance matrix
    """
    
    M = A.shape[0];                                                             # System dimension (N_particles + 1 cavity field)partículas  + 1 campo)
    
    A_T = np.transpose(A)                                                       # Transpose of the drift matrix
    
    V_old = np.reshape(V_old_vector, (M, M));                                      # Vector -> matrix
    
    dVdt = np.matmul(A, V_old) + np.matmul(V_old, A_T) + D;                     # Calculate how much the CM derivative in this time step
    
    dVdt_vector = np.reshape(dVdt, (M**2, 1));                                     # Matrix -> vector
    return dVdt_vector


class time_evolution:
    """
    Class simulating the time evolution of a gaussian state following a set of 
    Langevin and Lyapunov equations for its first moments dynamics
    
    ATTRIBUTES
        A                     - Drift matrix (can be a lambda functions to have a time dependency!)
        D                     - Diffusion Matrix 
        N                     - Mean values of the noises
        initial_state         - Initial state of the global system
        t                     - Array with timestamps for the time evolution
        
        is_stable             - Boolean telling if the system is stable or not
        R_semi_classical      - Array with semi-classical mean quadratures (Semi-classical time evolution using Monte Carlos method)
        R                     - Array with mean quadratures  for each time
        V                     - Cell  with covariance matrix for each time
        state                 - Gaussian state               for each time
                                                                                    
        N_time                - Length of time array
        Size_matrices         - Size of covariance, diffusion and drift matrices
        steady_state_internal - Steady state
    """
    
    def __init__(self, A_0, D_0, N_0, initial_state_0):
        """
        Class constructor for simulating the time evolution of the global system
        open/closed quantum dynamics dictated by Langevin and Lyapunov equations
        
        Langevin: \dot{R} = A*X + N           : time evolution of the mean quadratures
       
        Lyapunov: \dot{V} = A*V + V*A^T + D   : time evolution of the covariance matrix
       
        PARAMETERS:
           A_0           - Drift matrix     (numerical matrix or lambda functions for a matrix with time dependency
           D_0           - Diffusion Matrix (auto correlation of the noises, assumed to be delta-correlated in time)
           N_0           - Mean values of the noises
           initial_state - Cavity linewidth
       
        CALCULATES:
           self           - instance of a time_evolution class
           self.is_stable - boolean telling if the system is stable or not
        """
      
        self.A = A_0;                                                           # Drift matrix
        self.D = D_0;                                                           # Diffusion Matrix
        self.N = N_0;                                                           # Mean values of the noises
        
        self.initial_state = initial_state_0;                                   # Initial state of the global system
        
        self.Size_matrices = len(self.D);                                       # Size of system and ccupation number for the environment (heat bath)
      
        # assert 2*initial_state_0.N_modes == self.Size_matrices), "Initial state's number of modes does not match the drift and diffusion matrices sizes"              # Check if the initial state and diffusion/drift matrices have appropriate sizes !
      
        if( not is_lambda_function(self.A) ):
            eigvalue, eigvector = np.linalg.eig(self.A);                        # Calculate the eigenvalues of the drift matrix
            is_not_stable = np.any( eigvalue.real > 0 );                        # Check if any eigenvalue has positive real part (unstability)
            self.is_stable = not is_not_stable                                  # Store the information of the stability of the system in a class attribute
    
    def run(self, t_span):
        """
        Run every time evolution available at the input timestamps.
       
        PARAMETERS:
            self   - class instance
            tspan - Array with time stamps when the calculations should be done
       
        CALCULATES:
            result = array with time evolved gaussian states for each timestamp of the input argument t_span
            each entry of the array is a gaussian_state class instance
        """
      
        self.langevin(t_span);                                                  # Calculate the mean quadratures for each timestamp
      
        self.lyapunov(t_span);                                                  # Calculate the CM for each timestamp (perform time integration of the Lyapunov equation)
      
        self.build_states();                                                    # Combine the time evolutions calculated above into an array of gaussian states
      
        result = self.state;
        return result                                                           # Return the array of time evolved gaussian_state
    
    def langevin(self, t_span):
        """
        Solve the Langevin equation for the time evolved mean quadratures of the full system
       
        Uses ode45 to numerically integrate the average Langevin equations (a fourth order Runge-Kutta method)
       
        PARAMETERS:
            self   - class instance
            t_span - timestamps when the time evolution is to be calculated
       
        CALCULATES:
            self.R - a cell with the time evolved mean quadratures where
            self.R(i,j) is the i-th mean quadrature at the j-th timestamp
        """
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        if is_lambda_function(self.A):                                          # I have to check if there is a time_dependency on the odes :(
            langevin_ode = lambda t, R: np.matmul(self.A(t), R) + self.N        # Function handle that defines the Langevin equation (returns the derivative)
        else:
            langevin_ode = lambda t, R: np.matmul(self.A, R) + self.N           # Function handle that defines the Langevin equation (returns the derivative)
              
        solution_langevin = solve_ivp(langevin_ode, [t[0], t[-1]], self.initial_state.R, t_eval=t_span) # Solve Langevin eqaution through Runge Kutta(4,5)
        # Each row in R corresponds to the solution at the value returned in the corresponding row of self.t
      
        self.R = np.transpose(solution_langevin.y);                             # Store the time evolved quadratures in a class attribute
      
        #  fprintf("Langevin simulation finished!\n\n")                         # Warn user the heavy calculations ended
    
    def lyapunov(self, t_span):
        """
        Solve the lyapunov equation for the time evolved covariance matrix of the full system
       
        Uses ode45 to numerically integrate, a fourth order Runge-Kutta method
       
        PARAMETERS:
            self   - class instance
            t_span - timestamps when the time evolution is to be calculated
       
        CALCULATES:
            'self.V' - a cell with the time evolved covariance matrix where
             self.V{j} is the covariance matrix at the j-th timestamp
        """
      
        # disp("Lyapunov simulation started...")                                # Warn the user that heavy calculations started (their computer did not freeze!)
        
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = length(t_span);                                           # Number of timestamps
        
        V_0_vector = reshape(self.initial_state.V, [self.Size_matrices^2, 1]); # Reshape the initial condition into a vector (expected input for ode45)
        
        if is_lambda_function(self.A):                                          # I have to check if there is a time_dependency on the odes :(
            ode = lambda t, V: lyapunov_ode(t, V, self.A(t), self.D);           # Function handle that defines the Langevin equation (returns the derivative)
        else:
            ode = lambda t, V: lyapunov_ode(t, V, self.A, self.D);              # Function handle that defines the Langevin equation (returns the derivative)
        
        solution_langevin = solve_ivp(ode, [t[0], t[-1]], V_0_vector, t_eval=t_span) # Solve Lyapunov equation through Fourth order Runge Kutta
        
        # Unpack the output of ode45 into a cell where each entry contains the information about the evolved CM at each time
        self.V = [];                                                            # Initialize a cell to store all CMs for each time
        
        for i in range(len(V_vector)):
            V_current_vector = V_vector[i, :];                                  # Take the full Covariance matrix in vector form
            V_current = np.reshape(V_current_vector, (self.Size_matrices, self.Size_matrices)); # Reshape it into a proper matrix
            self.V.append(V_current);                                           # Append it on the class attribute
    
    def build_states(self):
        """
        Builds the gaussian state at each time from their mean values and covariance matrices
        This funciton is completely unnecessary, but it makes the code more readable :)
       
        CALCULATES:
          self.state - array with time evolved gaussian states for each timestamp of the input argument t_span
          each entry of the array is a gaussian_state class instance
        """
      
        assert self.R.size != 0 and self.V != 0, "No mean quadratures or covariance matrices, can not build time evolved states!"
        
        self.state = []
        
        for i in range(self.N_time):
            self.state.append( gaussian_state(self.R[:, i], self.V[i]) );
        
    def steady_state(self, A_0, A_c, A_s, omega):
        """
        Calculates the steady state for the system
       
        PARAMETERS:
          self   - class instance
        
          The next parameters are only necessary if the drift matrix has a time dependency (and it is periodic)
          A_0, A_c, A_s - components of the Floquet decomposition of the drift matrix
          omega - Frequency of the drift matrix
        
        CALCULATES:
          self.steady_state_internal with the steady state (gaussian_state)
          ss - gaussian_state with steady state of the system
        """
      
        if is_lambda_function(self.A):                                          # If the Langevin and Lyapunov eqs. have a time dependency, move to the Floquet solution
            ss = self.floquet(A_0, A_c, A_s, omega);
            self.steady_state_internal = ss;
        else :                                                                  # If the above odes are time independent, 
            assert self.is_stable, "There is no steady state covariance matrix, as the system is not stable!"  # Check if there exist a steady state!
        
        R_ss = np.linalg.solve(self.A, -self.N);                                # Calculate steady-state mean quadratures
        V_ss = solve_continuous_lyapunov(self.A, -self.D);                      # Calculate steady-state covariance matrix
        
        self.steady_state_internal = gaussian_state(R_ss, V_ss);                # Generate the steady state
        ss = self.steady_state_internal;                                        
        return ss                                                               # Return the gaussian_state with the steady state for this system
        return ss
    
    def floquet(self, A_0, A_c, A_s, omega):
        """
        Calculates the staeady state of a system with periodic Hamiltonin/drift matrix
        Uses first order approximation in Floquet space for this calculation
       
        Higher order approximations will be implemented in the future
        
        PARAMETERS:
          self   - class instance
        
          A_0, A_c, A_s - components of the Floquet decomposition of the drift matrix
          omega - Frequency of the drift matrix
        
        CALCULATES:
          self.steady_state_internal with the steady state (gaussian_state)
          ss - gaussian_state with steady state of the system
        """
      
        M = self.Size_matrices;                                                 # Size of the time-dependent matrix
        Id = np.identity(M);                                                    # Identity matrix for the system size
        
        A_F = np.block([[A_0,    A_c   ,     A_s  ],
                        [A_c,    A_0   , -omega*Id],
                        [A_s, +omega*Id,     A_0  ]])                           # Floquet drift     matrix
        
        D_F = np.kron(np.eye(3,dtype=int), self.D)                              # Floquet diffusion matrix
        
        N_F = np.vstack([self.N, self.N, self.N])                               # Floquet mean noise vector
        
        R_ss_F = np.linalg.solve(A_F, -N_F);                                    # Calculate steady-state Floquet mean quadratures vector
        V_ss_F = solve_continuous_lyapunov(A_F, -D_F);                          # Calculate steady-state Floquet covariance matrix
        
        R_ss = R_ss_F[1:M];                                                     # Get only the first entries
        V_ss = V_ss_F[1:M, 1:M];                                                # Get only the first sub-matrix
        
        self.steady_state_internal = gaussian_state(R_ss, V_ss); # Generate the steady state
        ss = self.steady_state_internal; 
        return ss
    
    def langevin_semi_classical(self, t_span, N_ensemble=2e+2):
        """
        Solve the semi-classical Langevin equation for the expectation value of the quadrature operators
        using a Monte Carlos simulation to numericaly integrate the Langevin equations
        
        The initial conditions follows the initial state probability density in phase space
        The differential stochastic equations are solved through a Euler-Maruyama method
       
        PARAMETERS:
          self   - class instance
          N_ensemble (optional) - number of iterations for Monte Carlos simulation, default value: 200
       
        CALCULATES:
          self.R_semi_classical - matrix with the quadratures expectation values of the time evolved system where 
          self.R_semi_classical(i,j) is the i-th quadrature expectation value at the j-th time
        """
      
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        dt = self.t(2) - self.t(1);                                             # Time step
        sq_dt =  np.sqrt(dt);                                                   # Square root of time step (for Wiener proccess in the stochastic integration)
        
        noise_amplitude = self.N + np.sqrt( np.diag(self.D) );                  # Amplitude for the noises (square root of the auto correlations)
        
        mean_0 = self.initial_state.R;                                          # Initial mean value
        std_deviation_0 =  np.sqrt( np.diag(self.initial_state.V) );            # Initial standard deviation
        
        self.R_semi_classical = np.zeros((self.Size_matrices, self.N_time));    # Matrix to store each quadrature ensemble average at each time
        
        if is_lambda_function(self.A):                                          # I have to check if there is a time_dependency on the odes
            AA = lambda t: self.A(t);                                           # Rename the function that calculates the drift matrix at each time
        else:
            AA = lambda t: self.A;                                              # If A is does not vary in time, the new function always returns the same value 
      
        for i in range(N_ensemble):                                             # Loop on the random initial positions (# Monte Carlos simulation using Euler-Maruyama method in each iteration)
            
            X = np.zeros((self.Size_matrices, self.N_time));                    # For this iteration, this matrix stores each quadrature at each time (first and second dimensions, respectively)
            X[:,1] = np.random.normal(mean_0, std_deviation_0)                  # Initial Cavity position quadrature (normal distribution for vacuum state)
            
            noise = np.random.standard_normal(X.shape);
            for k in range(self.N_time-1):                                      # Euler-Maruyama method for stochastic integration
                X[:,k+1] = X[:,k] + np.matmul(AA(self.t[k]), X[:,k])*dt + sq_dt*np.multiply(noise_amplitude, noise[:,k])
                                   
            self.R_semi_classical = self.R_semi_classical + X;                      # Add the new  Monte Carlos iteration quadratures to the same matrix
        
        self.R_semi_classical = self.R_semi_classical/N_ensemble;                 # Divide the ensemble sum to obtain the average quadratures at each time
      
        # fprintf("Langevin simulation ended\n")                                  # Warn user that heavy calculation started
    
    