# -*- coding: utf-8 -*-
"""
QuGIT - Quantum Gaussian Information Toolbox
Github: https://github.com/IgorBrandao42/Quantum-Gaussian-Information-Toolbox

Author: Igor Brandão
Contact: igorbrandao@aluno.puc-rio.br
"""


import numpy as np
from numpy.linalg import det
from numpy.linalg import matrix_power

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power


################################################################################


class gaussian_state:                                                           # Class definning a multimode gaussian state
    """Class simulation of a multimode gaussian state
    
    ATTRIBUTES:
        self.R       - Mean quadratures vector
        self.V       - Covariance matrix
        self.Omega   - Symplectic form matrix
        self.N_modes - Number of modes
    """    
    
    # Constructor and its auxiliar functions    
    def __init__(self, *args):
        """
        The user can explicitly pass the first two moments of a multimode gaussian state
        or pass a name-value pair argument to choose a single mode gaussian state
        
        PARAMETERS:
            R0, V0 - mean quadratures vector and covariance matrix of a gaussian state (ndarrays)
            
        NAME-VALUE PAIR ARGUMENTS:
            "vacuum"                        - generates vacuum   state (string)
            "thermal" , occupation number   - generates thermal  state (string, float)
            "coherent", complex amplitude   - generates coherent state (string, complex)
            "squeezed", squeezing parameter - generates squeezed state (string, float)
        """

        if(len(args) == 0):                                                     # Default constructor (vacuum state)
            self.R = np.array([[0], [0]])                                       # Save mean quadratres   in a class attribute
            self.V = np.identity(2)                                             # Save covariance matrix in a class attribute
            self.N_modes = 1;
             
        elif( isinstance(args[0], str) ):                                       # If the user called for an elementary gaussian state
            self.decide_which_state(args)                                       # Call the proper method to decipher which state the user wants 
        
        elif(isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray)): # If the user gave the desired mean quadratures values and covariance matrix
            R0 = args[0];
            V0 = args[1];
            
            R_is_real = all(np.isreal(R0))
            R_is_vector = np.squeeze(R0).ndim == 1
            
            V_is_matrix = np.squeeze(V0).ndim == 2
            V_is_square = V0.shape[0] == V0.shape[1]
            
            R_and_V_match = len(R0) == len(V0)
            
            assert R_is_real and R_is_vector and V_is_matrix and R_and_V_match and V_is_square, "Unexpected first moments when creating gaussian state!"  # Make sure they are a vector and a matrix with same length
        
            self.R = np.vstack(R0);                                             # Save mean quadratres   in a class attribute (vstack to ensure column vector)
            self.V = V0;                                                        # Save covariance matrix in a class attribute
            self.N_modes = int(len(R0)/2);                                           # Save the number of modes of the multimode state in a class attribute
            
        else:
            raise ValueError('Unexpected arguments when creating gaussian state!') # If input arguments do not make sense, call out the user
        
        omega = np.array([[0, 1], [-1, 0]]);                                    # Auxiliar variable
        self.Omega = np.kron(np.eye(self.N_modes,dtype=int), omega)             # Save the symplectic form matrix in a class attribute                                                    
    
    def decide_which_state(self, varargin):
        # If the user provided a name-pair argument to the constructor,
        # this function reads these arguments and creates the first moments of the gaussian state
      
        self.N_modes = 1;
        type_state = varargin[0];                                               # Name of expected type of gaussian state
      
        if(str(type_state) == "vacuum"):                                        # If it is a vacuum state
            self.R = np.array([[0], [0]])                                       # Save mean quadratres   in a class attribute
            self.V = np.identity(2)                                             # Save covariance matrix in a class attribute
            self.N_modes = 1;
            return                                                              # End function
      
                                                                                # Make sure there is an extra parameters that is a number
        assert len(varargin)>1, "Absent amplitude for non-vacuum elementary gaussian state"
        assert isinstance(varargin[1], (int, float, complex)), "Invalid amplitude for non-vacuum elementary gaussian state"
        
        if(str(type_state) == "thermal"):                                       # If it is a thermal state
            nbar = varargin[1];                                                 # Make sure its occuption number is a non-negative number
            assert nbar>=0, "Imaginary or negative occupation number for thermal state"
            self.R = np.array([[0], [0]])
            self.V = np.diag([2.0*nbar+1, 2.0*nbar+1]);                         # Create its first moments
        
        elif(str(type_state) == "coherent"):                                    # If it is a coherent state
           alpha = varargin[1];
           self.R = np.array([[2*alpha.real], [2*alpha.imag]]);
           self.V = np.identity(2);                                             # Create its first moments
        
        elif(str(type_state) == "squeezed"):                                    # If it is a squeezed state
            r = varargin[1];                                                    # Make sure its squeezing parameter is a real number
            assert np.isreal(r), "Unsupported imaginary amplitude for squeezed state"
            self.R = np.array([[0], [0]])
            self.V = np.diag([np.exp(-2*r), np.exp(+2*r)]);                           # Create its first moments
        
        else:
            self.N_modes = [];
            raise ValueError("Unrecognized gaussian state name, please check for typos or explicitely pass its first moments as arguments")
    
    def check_uncertainty_relation(self):
      """
      Check if the generated covariance matrix indeed satisfies the uncertainty principle (debbugging)
      """
      
      V_check = self.V + 1j*self.Omega;
      eigvalue, eigvector = np.linalg.eig(V_check)
      
      assert all(eigvalue>=0), "CM does not satisfy uncertainty relation!"
      
      return V_check
    
    def __str__(self):
        return str(self.N_modes) + "-mode gaussian state with mean quadrature vector R =\n" + str(self.R) + "\nand covariance matrix V =\n" + str(self.V)
    
    def copy(self):
        """Create identical copy"""
        
        return gaussian_state(self.R, self.V)
    
    # Construct another state, from this base gaussian_state
    def tensor_product(self, rho_list):
        """ Given a list of gaussian states, 
        # calculates the tensor product of the base state and the states in the array
        # 
        # PARAMETERS:
        #    rho_array - array of gaussian_state (multimodes)
        #
         CALCULATES:
            rho - multimode gaussian_state with all of the input states
        """
      
        R_final = self.R;                                                      # First moments of resulting state is the same of rho_A
        V_final = self.V;                                                      # First block diagonal entry is the CM of rho_A
      
        for rho in rho_list:                                                    # Loop through each state that is to appended
            R_final = np.vstack((R_final, rho.R))                               # Create its first moments
            V_final = block_diag(V_final, rho.V);
        
        temp = gaussian_state(R_final, V_final);                                 # Generate the gaussian state with these moments
        
        self.R = temp.R                                                         # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega   = temp.Omega
        self.N_modes = temp.N_modes
    
    def partial_trace(self, indexes):
        """
        Partial trace over specific single modes of the complete gaussian state
        
        PARAMETERS:
           indexes - the modes the user wants to trace out (as in the mathematical notation) 
        
        CALCULATES:
           rho_A - gaussian_state with all of the input state, except of the modes specified in 'indexes'
        """
      
        N_A = int(len(self.R) - 2*len(indexes));                                    # Twice the number of modes in resulting state
        assert N_A>=0, "Partial trace over more states than there exist in gaussian state" 
      
        # Shouldn't there be an assert over max(indexes) < obj.N_modes ? -> you cant trace out modes that do not exist
      
        modes = np.arange(self.N_modes)
        entries = np.isin(modes, indexes)
        entries = [not elem for elem in entries]
        modes = modes[entries];
      
        R0 = np.zeros((N_A, 1))
        V0 = np.zeros((N_A,N_A))
      
        for i in range(len(modes)):
            m = modes[i]
            R0[(2*i):(2*i+2)] = self.R[(2*m):(2*m+2)]
        
            for j in range(len(modes)):
                n = modes[j]
                V0[(2*i):(2*i+2), (2*j):(2*j+2)] = self.V[(2*m):(2*m+2), (2*n):(2*n+2)]
        
        temp = gaussian_state(R0, V0);                                          # Generate the gaussian state with these moments
        
        self.R = temp.R                                                         # Copy its attributes into the original instance
        self.V = temp.V
        self.Omega   = temp.Omega
        self.N_modes = temp.N_modes
    
    def only_modes(self, indexes):
      """
      Partial trace over all modes except the ones in indexes of the complete gaussian state
       
       PARAMETERS:
          indexes - the modes the user wants to retrieve from the multimode gaussian state
      
       CALCULATES:
          rho - gaussian_state with all of the specified modes
      """
      
      N_A = len(indexes);                                                       # Number of modes in resulting state
      assert N_A>0 and N_A <= self.N_modes, "Partial trace over more states than exists in gaussian state"
      
      R0 = np.zeros((2*N_A, 1))
      V0 = np.zeros((2*N_A, 2*N_A))
      
      for i in range(len(indexes)):
            m = indexes[i]
            R0[(2*i):(2*i+2)] = self.R[(2*m):(2*m+2)]
        
            for j in range(len(indexes)):
                n = indexes[j]
                V0[(2*i):(2*i+2), (2*j):(2*j+2)] = self.V[(2*m):(2*m+2), (2*n):(2*n+2)]
      
      temp = gaussian_state(R0, V0);                                            # Generate the gaussian state with these moments
        
      self.R = temp.R                                                           # Copy its attributes into the original instance
      self.V = temp.V
      self.Omega   = temp.Omega
      self.N_modes = temp.N_modes  
    
    def loss_ancilla(self,idx,tau):
        """
        Simulates a generic loss on mode idx by anexing an ancilla vacuum state and applying a
        beam splitter operator with transmissivity tau. The ancilla is traced-off from the final state. 
        
        PARAMETERS:
           idx - index of the mode that will suffer loss
           tau - transmissivity of the beam splitter
        
        CALCULATES:
            damped_state - final damped state
        """

        damped_state = tensor_product([self, gaussian_state("vacuum")])
        damped_state.beam_splitter(tau,[idx, damped_state.N_modes-1])
        damped_state.partial_trace([damped_state.N_modes-1])
        
        self.R = damped_state.R                                                 # Copy the damped state's attributes into the original instance
        self.V = damped_state.V
        self.Omega   = damped_state.Omega
        self.N_modes = damped_state.N_modes
    
    # Properties of the gaussian state
    def symplectic_eigenvalues(self):
        """
        Calculates the sympletic eigenvalues of a covariance matrix V with symplectic form Omega
        
        Finds the absolute values ofthe eigenvalues of i\Omega V and removes repeated entries
        
        CALCULATES:
            lambda - array with symplectic eigenvalues
        """  
        H = 1j*np.matmul(self.Omega, self.V);                                   # Auxiliar matrix
        lambda_0, v_0 = np.linalg.eig(H)
        lambda_0 = np.abs( lambda_0 );                                          # Absolute value of the eigenvalues of the auxiliar matrix
        
        lambda_s = np.zeros((self.N_modes, 1));                                 # Variable to store the symplectic eigenvalues
        for i in range(self.N_modes):                                           # Loop over the non-repeated entries of lambda_0
            lambda_s[i] = lambda_0[0]                                         # Get the first value on the repeated array
            lambda_0 = np.delete(lambda_0, 0)                                  # Delete it
            
            idx = np.argmin( np.abs(lambda_0-lambda_s[i]) )                           # Find the next closest value on the array (repeated entry)
            lambda_0 = np.delete(lambda_0, idx)                              # Delete it too
        
        return lambda_s
    
    def purity(self):
      """
      Purity of a gaussian state (pure states have unitary purity)
       
       CALCULATES:
           p - purity
      """
      
      return 1/np.prod( self.symplectic_eigenvalues() );
    
    def squeezing_degree(self):
        """
        Degree of squeezing of the quadratures of a single mode state
        Defined as the ratio of the variance of the squeezed and antisqueezed quadratures
        
        CALCULATES:
            eta   - ratio of the variances above
            V_sq  - variance of the     squeezed quadrature
            V_asq - variance of the antisqueezed quadrature
                   
        REFERENCE: 
            Phys. Rev. Research 2, 013052 (2020)
        """
      
        assert self.N_modes == 1, "At the moment, this program only calculates the squeezing degree for a single mode state"
      
        lambda_0, v_0 = np.linalg.eig(self.V)
        
        V_sq  = np.amin(lambda_0);
        V_asq = np.amax(lambda_0);
      
        eta = V_sq/V_asq;
        return eta, V_sq, V_asq
    
    def von_Neumann_Entropy(self):
        """
        Calculation of the von Neumann entropy for a multipartite gaussian system
       
        CALCULATES:
             Entropy - von Neumann entropy of the multimode state
        """
        
        nu = self.symplectic_eigenvalues();                                     # Calculates the sympletic eigenvalues of a covariance matrix V
        
                                                                                # 0*log(0) is NaN, but in the limit that x->0 : x*log(x) -> 0
        # nu[np.abs(nu - 1) < 1e-15] = nu[np.abs(nu - 1) < 1e-15] + 1e-15;                                 # Doubles uses a 15 digits precision, I'm adding a noise at the limit of the numerical precision
        nu[np.abs(nu-1) < 1e-15] = 1+1e-15
        
        nu_plus  = (nu + 1)/2.0;                                                # Temporary variables
        # nu_minus = (nu - 1)/2.0;
        nu_minus = np.abs((nu - 1)/2.0);
        g_nu = np.multiply(nu_plus,np.log(nu_plus)) - np.multiply(nu_minus, np.log(nu_minus))
      
        Entropy = np.sum( g_nu );                                               # Calculate the entropy
        return Entropy
    
    def mutual_information(self):
        """
         Mutual information for a multipartite gaussian system
        
         CALCULATES:
            I     - mutual information  for the total system of the j-th covariance matrix
            S_tot - von Neumann entropy for the total system of the j-th covariance matrix
            S     - von Neumann entropy for the i-th mode    of the j-th covariance matrix
        """
        S = np.zeros((self.N_modes, 1));                                        # Variable to store the entropy of each mode
        
        for j in range(self.N_modes):                                           # Loop through each mode
            single_mode = only_modes(self, [j]);                                # Get the covariance matrix for only the i-th mode
            S[j] = single_mode.von_Neumann_Entropy();                           # von Neumann Entropy for i-th mode of each covariance matrix
        
        S_tot = self.von_Neumann_Entropy();                                     # von Neumann Entropy for the total system of each covariance matrix
        
        I = np.sum(S) - S_tot;                                                  # Calculation of the mutual information
        return I
    
    def occupation_number(self):
        """
        Occupation number for a each single mode within the multipartite gaussian state (array)
        
        CALCULATES:
            nbar - array with the occupation number for each single mode of the multipartite gaussian state
        """
        
        Variances = np.diag(self.V);                                                # From the current CM, take take the variances
        Variances = np.vstack(Variances)
        
        mean_x = self.R[::2];                                                    # Odd  entries are the mean values of the position
        mean_p = self.R[1::2];                                                   # Even entries are the mean values of the momentum
        
        Var_x = Variances[::2];                                                 # Odd  entries are position variances
        Var_p = Variances[1::2];                                                # Even entries are momentum variances
        
        nbar = 0.25*( Var_x + mean_x**2 + Var_p + mean_p**2 ) - 0.5;            # Calculate occupantion numbers at current time
        return nbar
    
    def number_operator_moments(self):
        """
        Calculates means vector and covariance matrix of photon numbers for each mode of the gaussian state
        
        CALCULATES:
            m - mean values of number operator in arranged in a vector (Nx1 numpy.ndarray)
            K - covariance matrix of the number operator               (NxN numpy.ndarray)
           
        REFERENCE:
            Phys. Rev. A 99, 023817 (2019)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        q = self.R[::2]                                                         # Mean values of position quadratures (even entries of self.R)
        p = self.R[1::2]                                                        # Mean values of momentum quadratures (odd  entries of self.R)
        
        alpha   = 0.5*(q + 1j*p)                                                # Mean values of annihilation operators
        alpha_c = 0.5*(q - 1j*p)                                                # Mean values of creation     operators
        
        V_1 = self.V[0::2, 0::2]/2.0                                            # Auxiliar matrix
        V_2 = self.V[0::2, 1::2]/2.0                                            # Auxiliar matrix
        V_3 = self.V[1::2, 1::2]/2.0                                            # Auxiliar matrix
        
        A = ( V_1 + V_3 + 1j*(np.transpose(V_2) - V_2) )/2.0                    # Auxiliar matrix
        B = ( V_1 - V_3 + 1j*(np.transpose(V_2) + V_2)   )/2.0                    # Auxiliar matrix
        
        temp = np.multiply(np.matmul(alpha_c, alpha.transpose()), A) + np.multiply(np.matmul(alpha_c, alpha_c.transpose()), B) # Yup, you guessed it, another auxiliar matrix
        
        m = np.real(np.reshape(np.diag(A), (self.N_modes,1)) + np.multiply(alpha, alpha_c) - 0.5) # Mean values of number operator (occupation numbers)
        
        K = np.real(np.multiply(A, A.conjugate()) + np.multiply(B, B.conjugate()) - 0.25*np.eye(self.N_modes)  + 2.0*temp.real) # Covariance matrix for the number operator
        
        return m, K
    
    def coherence(self):
        """
        Coherence of a multipartite gaussian system
         
        CALCULATES:
            C - coherence
        
        REFERENCE: 
            Phys. Rev. A 93, 032111 (2016).
        """
        
        nbar = self.occupation_number();                                        # Array with each single mode occupation number
        
        nbar[nbar==0] = nbar[nbar==0] + 1e-16;                                  # Make sure there is no problem with log(0)!
        
        S_total = self.von_Neumann_Entropy();                                    # von Neumann Entropy for the total system
        
        temp = np.sum( np.multiply(nbar+1, np.log2(nbar+1)) - np.multiply(nbar, np.log2(nbar)) );                # Temporary variable
        
        C = temp - S_total;                                                     # Calculation of the mutual information
        return C
    
    def logarithmic_negativity(self, *args):
        """
        Calculation of the logarithmic negativity for a bipartite system
       
        PARAMETERS:
           indexes - array with indices for the bipartition to consider 
           If the system is already bipartite, this parameter is optional !
       
        CALCULATES:
           LN - logarithmic negativity for the bipartition / bipartite states
        """
        
        temp = self.N_modes 
        if(temp == 2):                                                          # If the full system is only comprised of two modes
            V0 = self.V                                                         # Take its full covariance matrix
        elif(len(args) > 0 and temp > 2):
            indexes = args[0]
            
            assert len(indexes) == 2, "Can only calculate the logarithmic negativity for a bipartition!"
                
            bipartition = only_modes(self,indexes)                              # Otherwise, get only the two mode specified by the user
            V0 = bipartition.V                                                  # Take the full Covariance matrix of this subsystem
        else:
            raise TypeError('Unable to decide which bipartite entanglement to infer, please pass the indexes to the desired bipartition')
        
        A = V0[0:2, 0:2]                                                        # Make use of its submatrices
        B = V0[2:4, 2:4] 
        C = V0[0:2, 2:4] 
        
        sigma = np.linalg.det(A) + np.linalg.det(B) - 2.0*np.linalg.det(C)      # Auxiliar variable
        
        ni = sigma/2.0 - np.sqrt( sigma**2 - 4.0*np.linalg.det(V0) )/2.0 ;      # Square of the smallest of the symplectic eigenvalues of the partially transposed covariance matrix
        
        if(ni < 0.0):                                                           # Manually perform a maximum to save computational time (calculation of a sqrt can take too much time and deal with residual numeric imaginary parts)
            LN = 0.0;
        else:
            ni = np.sqrt( ni.real );                                            # Smallest of the symplectic eigenvalues of the partially transposed covariance matrix
        
        LN = np.max([0, -np.log(ni)]);                                          # Calculate the logarithmic negativity at each time
        return LN
    
    def fidelity(self, rho_2):
        """
        Calculates the fidelity between the two arbitrary gaussian states
        
        ARGUMENTS:
            rho_1, rho_2 - gaussian states to be compared through fidelity
         
        CALCULATES:
            F - fidelity between rho_1 and rho_2
        
        REFERENCE:
            Phys. Rev. Lett. 115, 260501.
       
        OBSERVATION:
        The user should note that non-normalized quadratures are expected;
        They are normalized to be in accordance with the notation of Phys. Rev. Lett. 115, 260501.
        """
      
        assert self.N_modes == rho_2.N_modes, "Impossible to calculate the fidelity between gaussian states of diferent sizes!" 
        
        u_1 = self.R/np.sqrt(2.0);                                              # Normalize the mean value of the quadratures
        u_2 = rho_2.R/np.sqrt(2.0);
        
        V_1 = self.V/2.0;                                                       # Normalize the covariance matrices
        V_2 = rho_2.V/2.0;
        
        OMEGA = self.Omega;
        OMEGA_T = np.transpose(OMEGA)
        
        delta_u = u_2 - u_1;                                                    # A bunch of auxiliar variables
        delta_u_T = np.hstack(delta_u)
        
        inv_V = np.linalg.inv(V_1 + V_2);
        
        V_aux = np.matmul( np.matmul(OMEGA_T, inv_V), OMEGA/4 + np.matmul(np.matmul(V_2, OMEGA), V_1) )
        
        identity = np.identity(2*self.N_modes);
        
        # V_temp = np.linalg.pinv(np.matmul(V_aux,OMEGA))                         # Trying to bypass singular matrix inversion ! I probably shouldnt do this...
        # F_tot_4 = np.linalg.det( 2*np.matmul(sqrtm(identity + matrix_power(V_temp                ,+2)/4) + identity, V_aux) );
        F_tot_4 = np.linalg.det( 2*np.matmul(sqrtm(identity + matrix_power(np.matmul(V_aux,OMEGA),-2)/4) + identity, V_aux) );
        
        F_0 = (F_tot_4.real / np.linalg.det(V_1+V_2))**(1.0/4.0);               # We take only the real part of F_tot_4 as there can be a residual complex part from numerical calculations!
        
        F = F_0*np.exp( -np.matmul(np.matmul(delta_u_T,inv_V), delta_u)  / 4);                        # Fidelity
        return F
    
    # Gaussian unitaries (applicable to single mode states)
    def displace(self, alpha, modes=[0]):
        """
        Apply displacement operator
       
        ARGUMENT:
           alpha - complex amplitudes for the displacement operator
           modes - indexes for the modes to be displaced 
        """
        
        if not (isinstance(alpha, list) or isinstance(alpha, np.ndarray) or isinstance(alpha, range)):      # Make sure the input variables are of the correct type
            alpha = [alpha]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(alpha), "Unable to decide which modes to displace nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        for i in range(len(alpha)):                                             # For each displacement amplitude
            idx = modes[i]                                                      # Get its corresponding mode
            
            d = 2.0*np.array([[alpha[i].real], [alpha[i].imag]]);               # Discover by how much this mode is to be displaced
            self.R[2*idx:2*idx+2] = self.R[2*idx:2*idx+2] + d;                  # Displace its mean value (covariance matrix is not altered)
    
    def squeeze(self, r, modes=[0]):
        """
        Apply squeezing operator on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           r     - ampllitude for the squeezing operator
           modes - indexes for the modes to be squeezed
        """
        
        if not (isinstance(r, list) or isinstance(r, np.ndarray) or isinstance(r, range)):              # Make sure the input variables are of the correct type
            r = [r]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(r), "Unable to decide which modes to squeeze nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        S = np.eye(2*self.N_modes)                                              # Build the squeezing matrix (initially a identity matrix because there is no squeezing to be applied on other modes)
        for i in range(len(r)):                                                 # For each squeezing parameter
            idx = modes[i]                                                      # Get its corresponding mode
            
            S[2*idx:2*idx+2, 2*idx:2*idx+2] = np.diag([np.exp(-r[i]), np.exp(+r[i])]); # Build the submatrix that squeezes the desired modes
        
        self.R = np.matmul(S, self.R);                                          # Apply squeezing operator on first  moments
        self.V = np.matmul( np.matmul(S,self.V), S);                            # Apply squeezing operator on second moments
        
    def rotate(self, theta, modes=[0]):
        """
        Apply phase rotation on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           theta - ampllitude for the rotation operator
           modes - indexes for the modes to be squeezed
        """
        
        if not (isinstance(theta, list) or isinstance(theta, np.ndarray) or isinstance(theta, range)):      # Make sure the input variables are of the correct type
            theta = [theta]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == len(theta), "Unable to decide which modes to rotate nor by how much" # If the size of the inputs are different, there is no way of telling exactly what it is expected to do
        
        Rot = np.eye(2*self.N_modes)                                            # Build the rotation matrix (initially identity matrix because there is no rotation to be applied on other modes)
        for i in range(len(theta)):                                             # For each rotation angle
            idx = modes[i]                                                      # Get its corresponding mode
            
            Rot[2*idx:2*idx+2, 2*idx:2*idx+2] = np.array([[np.cos(theta[i]), np.sin(theta[i])], [-np.sin(theta[i]), np.cos(theta[i])]]); # Build the submatrix that rotates the desired modes
        
        Rot_T = np.transpose(Rot)
        
        self.R = np.matmul(Rot, self.R);                                        # Apply rotation operator on first  moments
        self.V = np.matmul( np.matmul(Rot, self.V), Rot_T);                     # Apply rotation operator on second moments
        
    def phase(self, theta, modes=[0]):
        """
        Apply phase rotation on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           theta - ampllitude for the rotation operator
           modes - indexes for the modes to be squeezed
        """
        self.rotate(theta, modes)                                               # They are the same method/operator, this is essentially just a alias
    
    # Gaussian unitaries (applicable to two mode states)
    def beam_splitter(self, tau, modes=[0, 1]):
        """
        Apply a beam splitter transformation to pair of modes in a multimode gaussian state
        
        ARGUMENT:
           tau   - transmissivity of the beam splitter
           modes - indexes for the pair of modes which will receive the beam splitter operator 
        """
        
        # if not (isinstance(tau, list) or isinstance(tau, np.ndarray)):          # Make sure the input variables are of the correct type
        #     tau = [tau]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == 2, "Unable to decide which modes to apply beam splitter operator nor by how much"
        
        BS = np.eye(2*self.N_modes)
        i = modes[0]
        j = modes[1] 
        
        # B = np.sqrt(tau)*np.identity(2)
        # S = np.sqrt(1-tau)*np.identity(2)
        
        # BS[2*i:2*i+2, 2*i:2*i+2] = B
        # BS[2*j:2*j+2, 2*j:2*j+2] = B
        
        # BS[2*i:2*i+2, 2*j:2*j+2] =  S
        # BS[2*j:2*j+2, 2*i:2*i+2] = -S
        
        ##########################################
        sin_theta = np.sqrt(tau)
        cos_theta = np.sqrt(1-tau)
        
        BS[2*i  , 2*i  ] = sin_theta
        BS[2*i+1, 2*i+1] = sin_theta
        BS[2*j  , 2*j  ] = sin_theta
        BS[2*j+1, 2*j+1] = sin_theta
        
        BS[2*i+1, 2*j  ] = +cos_theta
        BS[2*j+1, 2*i  ] = +cos_theta
        
        BS[2*i  , 2*j+1] = -cos_theta
        BS[2*j  , 2*i+1] = -cos_theta
        ##########################################
        
        BS_T = np.transpose(BS)
        
        self.R = np.matmul(BS, self.R);
        self.V = np.matmul( np.matmul(BS, self.V), BS_T);
    
    def two_mode_squeezing(self, r, modes=[0, 1]):
        """
        Apply a two mode squeezing operator  in a gaussian state
        r - squeezing parameter
        
        ARGUMENT:
           r - ampllitude for the two-mode squeezing operator
        """
        
        # if not (isinstance(r, list) or isinstance(r, np.ndarray)):              # Make sure the input variables are of the correct type
        #     r = [r]
        if not (isinstance(modes, list) or isinstance(modes, np.ndarray) or isinstance(modes, range)):      # Make sure the input variables are of the correct type
            modes = [modes]
        
        assert len(modes) == 2, "Unable to decide which modes to apply two-mode squeezing operator nor by how much"
        
        S2 = np.eye(2*self.N_modes)
        i = modes[0]
        j = modes[1] 
        
        S0 = np.cosh(r)*np.identity(2);
        S1 = np.sinh(r)*np.diag([+1,-1]);
        
        S2[2*i:2*i+2, 2*i:2*i+2] = S0
        S2[2*j:2*j+2, 2*j:2*j+2] = S0
        
        S2[2*i:2*i+2, 2*j:2*j+2] = S1
        S2[2*j:2*j+2, 2*i:2*i+2] = S1
        
        # S2 = np.block([[S0, S1], [S1, S0]])
        S2_T = np.transpose(S2)
        
        self.R = np.matmul(S2, self.R);
        self.V = np.matmul( np.matmul(S2, self.V), S2_T)
        
    # Generic multimode gaussian unitary
    def apply_unitary(self, S, d):
        """
        Apply a generic gaussian unitary on the gaussian state
        
        ARGUMENTS:
            S,d - affine symplectic map (S, d) acting on the phase space, equivalent to gaussian unitary
        """
        assert all(np.isreal(d)) , "Error when applying generic unitary, displacement d is not real!"
        
        S_is_symplectic = np.allclose(np.matmul(np.matmul(S, self.Omega), S.transpose()), self.Omega)
        
        assert S_is_symplectic , "Error when applying generic unitary, unitary S is not symplectic!"
        
        self.R = np.matmul(S, self.R) + d
        self.V = np.matmul(np.matmul(S, self.V), S.transpose())
        
    # Gaussian measurements
    def measurement_general(self, *args):
        """
        After a general gaussian measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or its mean value and covariance matrix
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           R_m      - first moments     of the conditional state after the measurement
           V_m      - covariance matrix of the conditional state after the measurement
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_B.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
           Conditional and unconditional Gaussian quantum dynamics - Contemp. Phys. 57, 331 (2016)
        """
        if isinstance(args[0], gaussian_state):                                 # If the input argument is a gaussian_state
            R_m   = args[0].R;
            V_m   = args[0].V;
            rho_m = args[0]
        else:                                                                   # If the input arguments are the conditional state's mean quadrature vector anc covariance matrix
            R_m = args[0];
            V_m = args[1];
            rho_m = gaussian_state(R_m, V_m)
        
        idx_modes = range(int(self.N_modes-len(R_m)/2), self.N_modes);          # Indexes to the modes that are to be measured
        
        rho_B = only_modes(self, idx_modes);                                    # Get the mode measured mode in the global state previous to the measurement
        rho_A = partial_trace(self, idx_modes);                                 # Get the other modes in the global state        previous to the measurement
        
        n = 2*rho_A.N_modes;                                                    # Twice the number of modes in state A
        m = 2*rho_B.N_modes;                                                    # Twice the number of modes in state B
        
        V_AB = self.V[0:n, n:(n+m)];                                            # Get the matrix dictating the correlations      previous to the measurement                           
        
        inv_aux = np.linalg.inv(rho_B.V + V_m)                                  # Auxiliar variable
        
        # Update the other modes conditioned on the measurement results
        rho_A.R = rho_A.R - np.matmul(V_AB, np.linalg.solve(rho_B.V + V_m, rho_B.R - R_m) );
        
        rho_A.V = rho_A.V - np.matmul(V_AB, np.matmul(inv_aux, V_AB.transpose()) );
        
        rho_A.tensor_product([rho_m])                                           # Generate the post measurement gaussian state
        
        self.R = rho_A.R                                                        # Copy its attributes into the original instance
        self.V = rho_A.V
        self.Omega   = rho_A.Omega
        self.N_modes = rho_A.N_modes
    
    def measurement_homodyne(self, *args):
        """
        After a homodyne measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or its mean quadrature vector
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           R_m      - first moments of the conditional state after the measurement (assumes measurement on position quadrature
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_B.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
        """
      
        if isinstance(args[0], gaussian_state):                                 # If the input argument is a gaussian_state
            R_m   = args[0].R;
            rho_m = args[0]
        else:                                                                   # If the input argument is the mean quadrature vector
            R_m = args[0];
            V_m = args[1];
            rho_m = gaussian_state(R_m, V_m)
        
        idx_modes = range(int(self.N_modes-len(R_m)/2), self.N_modes);          # Indexes to the modes that are to be measured
        
        rho_B = only_modes(self, idx_modes);                                    # Get the mode measured mode in the global state previous to the measurement
        rho_A = partial_trace(self, idx_modes);                                 # Get the other modes in the global state        previous to the measurement
        
        n = 2*rho_A.N_modes;                                                    # Twice the number of modes in state A
        m = 2*rho_B.N_modes;                                                    # Twice the number of modes in state B
        
        V_AB = self.V[0:n, n:(n+m)];                                            # Get the matrix dictating the correlations      previous to the measurement
        
        MP_inverse = np.diag([1/rho_B.V[1,1], 0]);                              # Moore-Penrose pseudo-inverse an auxiliar matrix (see reference)
        
        rho_A.R = rho_A.R - np.matmul(V_AB, np.matmul(MP_inverse, rho_B.R - R_m   ) ); # Update the other modes conditioned on the measurement results
        rho_A.V = rho_A.V - np.matmul(V_AB, np.matmul(MP_inverse, V_AB.transpose()) );
        
        rho_A.tensor_product([rho_m])                                           # Generate the post measurement gaussian state
        
        self.R = rho_A.R                                                        # Copy its attributes into the original instance
        self.V = rho_A.V
        self.Omega   = rho_A.Omega
        self.N_modes = rho_A.N_modes
    
    def measurement_heterodyne(self, *args):
        """
        After a heterodyne measurement is performed on the last m modes of a (n+m)-mode gaussian state
        this method calculates the conditional state the remaining n modes evolve into
        
        The user must provide the gaussian_state of the measured m-mode state or the measured complex amplitude of the resulting coherent state
        
        At the moment, this method can only perform the measurement on the last modes of the global state,
        if you know how to perform this task on a generic mode, contact me so I can implement it! :)
       
        ARGUMENTS:
           alpha    - complex amplitude of the coherent state after the measurement
           or
           rho_m    - conditional gaussian state after the measurement on the last m modes (rho_m.N_modes = m)
        
        REFERENCE:
           Jinglei Zhang's PhD Thesis - https://phys.au.dk/fileadmin/user_upload/Phd_thesis/thesis.pdf
        """
        
        if isinstance(args[0], gaussian_state):                                 # If the input argument is  a gaussian_state
            rho_m = args[0];
        else:
            rho_m = gaussian_state("coherent", args[0]);
        
        self.measurement_general(rho_m);
        
    
    # Phase space representation
    def wigner(self, X, P):
        """
        Calculates the wigner function for a single mode gaussian state
       
        PARAMETERS
            X, P - 2D grid where the wigner function is to be evaluated (use meshgrid)
        
        CALCULATES:
            W - array with Wigner function over the input 2D grid
        """
        
        assert self.N_modes == 1, "At the moment, this program only calculates the wigner function for a single mode state"
        
        N = self.N_modes;                                                       # Number of modes
        W = np.zeros((len(X), len(P)));                                         # Variable to store the calculated wigner function
        
        one_over_purity = 1/self.purity();
        
        inv_V = np.linalg.inv(self.V)
        
        for i in range(len(X)):
            x = np.block([ [X[i,:]] , [P[i,:]] ]);   
            
            for j in range(x.shape[1]):
                dx = np.vstack(x[:, j]) - self.R;                                          # x_mean(:,i) is the i-th point in phase space
                dx_T = np.hstack(dx)
                
                W_num = np.exp( - np.matmul(np.matmul(dx_T, inv_V), dx)/2 );    # Numerator
                
                W_den = (2*np.pi)**N * one_over_purity;                         # Denominator
          
                W[i, j] = W_num/W_den;                                          # Calculate the wigner function at every point on the grid
        return W
    
    def q_function(self, *args):
        """
        Calculates the Hussimi Q-function over a meshgrid
        
        PARAMETERS (numpy.ndarray, preferably generated by np.meshgrid):
            X, Y - 2D real grid where the Q-function is to be evaluated (use meshgrid to generate the values on the axes)
            OR
            ALPHA - 2D comples grid, each entry on this matrix is a vertex on the grid (equivalent to ALPHA = X + 1j*Y)
        
        CALCULATES:
            q_func - array with q-function over the input 2D grid
           
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        
        # Handle input into correct form
        if len(args) > 1:                                                      # If user passed more than one argument (should be X and Y - real values of on the real and imaginary axes)
            X = args[0]
            Y = args[1]
            ALPHA = X + 1j*Y                                                    # Then, construct the complex grid
        else:
            ALPHA = args[0]                                                     # If the user passed a single argument, it should be the complex grid, just rename it
        
        ALPHA = np.array(ALPHA)                                                 # Make sure ALPHA is the correct type
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = one_over_sqrt_2*np.block([[-1j*eye_N, +1j*eye_N],
                                      [    eye_N,     eye_N]]);                 # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        # Loop through the meshgrid and evaluate Q-function
        q_func = np.zeros(ALPHA.shape)
        
        for i in range(ALPHA.shape[0]):
            for j in range(ALPHA.shape[1]):
        
                gamma = np.zeros(2*self.N_modes,dtype=np.complex_)              # Auxiliar 2*self.N_modes complex vector      
                gamma[:self.N_modes] = np.conj(ALPHA[i, j])                     # First N entries are the complex conjugate of alpha
                gamma[self.N_modes:] = ALPHA[i, j]                              # Last  N entries are alpha
                
                q_func[i,j] = np.real(P_0*np.exp( -0.5*np.matmul(np.conj(gamma),gamma) -0.5*np.matmul( gamma , np.matmul(R,gamma)) + np.matmul( gamma , np.matmul(R,y)) ))
        
        q_func = q_func / (np.pi**self.N_modes)
        
        return q_func
    
    # Density matrix elements
    def density_matrix_coherent_basis(self, alpha, beta):
        """
        Calculates the matrix elements of the density operator on the coherent state basis
        
        PARAMETERS:
            alpha - a N-array with complex aplitudes (1xN numpy.ndarray)
            beta - a N-array with complex aplitudes (NxN numpy.ndarray)
        
        CALCULATES:
            q_f - the matrix element \bra{\alpha}\rho\kat{\beta}
           
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
            Many thanks to Daniel Tandeitnik for the base code for this method!
        """
        
        assert (len(alpha) == len(beta)) and (len(alpha) == self.N_modes), "Wrong input dimensions for the matrix element of the density matrix in coherent state basis!"
        
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = np.block([[-1j*one_over_sqrt_2*eye_N, +1j*one_over_sqrt_2*eye_N],
                      [    one_over_sqrt_2*eye_N,     one_over_sqrt_2*eye_N]]); # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros(2*self.N_modes)                                            # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        gamma = np.zeros(2*self.N_modes,dtype=np.complex_)                      # Auxiliar 2*self.N_modes complex vector      
        gamma[:self.N_modes] = np.conj(beta)                                    # First N entries are the complex conjugate of beta
        gamma[self.N_modes:] = alpha                                            # Last  N entries are alpha
        
        beta_rho_alpha = P_0*np.exp( -0.5*np.matmul(np.conj(gamma),gamma) -0.5*np.matmul( gamma , np.matmul(R,gamma)) + np.matmul( gamma , np.matmul(R,y)) ) # Hussimi Q-function
        
        return beta_rho_alpha
    
    def density_matrix_number_basis(self, n_cutoff=10):
        """
        Calculates the number distribution of the gaussian state
        
        PARAMETERS:
            n_cutoff - maximum number for the calculation
            
        RETURNS:
            P - array with the number distribution of the state (P.shape = self.N_modes*[n_cutoff])
            
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
        """
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2)                                        # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = np.block([[-1j*one_over_sqrt_2*eye_N, +1j*one_over_sqrt_2*eye_N],
                      [    one_over_sqrt_2*eye_N,     one_over_sqrt_2*eye_N]]); # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
        
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable        
        
        H = Hermite_multidimensional(R, y, n_cutoff)                            # Calculate the Hermite polynomial associated with this gaussian state
        
        # Calculate the probabilities
        rho_m_n = np.zeros((2*self.N_modes)*[n_cutoff])                         # Initialize the tensor to 0 (n_cutoff entries in each of the 2*self.N_modes dimensions)
        
        # rho is the same shape as H !
        
        m_last = np.array((2*self.N_modes)*[0], dtype=int)
        idx = np.ravel_multi_index(list(m_last), dims=rho_m_n.shape, order='F') # Get "linearized" index
        rho_m_n.ravel()[idx] = P_0                                              # Set its first entry to P_0

        # Similar procedure to what precedes. Move forward in the P tensor and fill it element by element.
        # next_m = np.ones([self.N_modes, 1], dtype=int);
        # next_n = np.ones([self.N_modes, 1], dtype=int);
        
        n_entries = np.prod(H.shape)                                            # Number of entries on the multidimensional Hermite polynomial tensor
        
        for mn_linear in range(0, n_entries):                                   # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
            mn = np.array(np.unravel_index(mn_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
            
            m = mn[:self.N_modes]                                               # First self.N_modes entries are the vector m
            n = mn[self.N_modes:]                                               # Last  self.N_modes entries are the vector n
            
            big_factorial = 1.0
            for kk in range(self.N_modes):                                      # Next, divide by the square root of the appropriate factorial! # kk = 1:dim/2
                big_factorial = big_factorial*np.math.factorial(m[kk])*np.math.factorial(n[kk]);
            
            rho_m_n.ravel()[mn_linear] = P_0*H.ravel()[mn_linear]/np.sqrt(big_factorial)
                    
        return rho_m_n 
    
    def number_statistics(self, n_cutoff=10):
        """
        Calculates the number distribution of the gaussian state
        
        PARAMETERS:
            n_cutoff - maximum number for the calculation
            
        RETURNS:
            P - array with the number distribution of the state (P.shape = self.N_modes*[n_cutoff])
            
        REFERENCE:
            Phys. Rev. A 50, 813 (1994)
        """
        
        # Preamble, get auxiliar variables that depend only on the gaussian state parameters
        one_over_sqrt_2 = 1.0/np.sqrt(2.0)                                      # Auxiliar variable to save computation time
        
        eye_N  = np.eye(  self.N_modes)                                         # NxN   identity matrix (auxiliar variable to save computation time)
        eye_2N = np.eye(2*self.N_modes)                                         # 2Nx2N identity matrix (auxiliar variable to save computation time)
        
        U = one_over_sqrt_2*np.block([[-1j*eye_N, +1j*eye_N],
                                      [    eye_N,     eye_N]]);                 # Auxiliar unitary matrix
        
        M = np.block([[      self.V[1::2, 1::2]        , self.V[1::2, 0::2]],   # Covariance matrix in new notation
                      [np.transpose(self.V[1::2, 0::2]), self.V[0::2, 0::2]]])/2.0;
        
        if np.allclose(2.0*M, eye_2N, rtol=1e-14, atol=1e-14):                      # If the cavirance matrix is the identity matrix, there will numeric errors below,
            M = (1-1e-15)*M                                                     # We circumvent this by adding a noise on the last digit of a floating point number
            
        Q = np.zeros([2*self.N_modes,1])                                        # Mean quadrature vector (rearranged)
        Q[:self.N_modes] = one_over_sqrt_2*self.R[1::2]                         # First self.N_modes entries are mean position quadratures
        Q[self.N_modes:] = one_over_sqrt_2*self.R[::2]                          # Last  self.N_modes entries are mean momentum quadratures
        
        Q_T = np.reshape(Q, [1, len(Q)])                                        # Auxiliar vector (transpose of vector Q)
        aux_inv = np.linalg.pinv(eye_2N + 2.0*M)                                # Auxiliar matrix (save time only inverting a single time!)
        
        R = np.matmul( np.matmul( U.conj().transpose() , eye_2N-2.0*M) , np.matmul( aux_inv , U.conj() ) ) # Auxiliar variable
        y = 2.0*np.matmul( np.matmul( U.transpose() , np.linalg.pinv(eye_2N-2.0*M) ) , Q )                 # Auxiliar variable
        P_0 = ( det(M + 0.5*eye_2N)**(-0.5) )*np.exp( -np.matmul( Q_T , np.matmul( aux_inv , Q )  ) )      # Auxiliar variable
        
        # DEBUGGING !
        # R_old = np.matmul( np.matmul( np.conj(np.transpose(U)) , (1+1e-15)*eye_2N-2*M) , np.matmul( np.linalg.pinv(eye_2N+2*M) , np.conj(U) ) ) # Auxiliar variable
        # y_old = 2*np.matmul( np.matmul( np.transpose(U) , np.linalg.pinv((1+1e-15)*eye_2N-2*M) ) , Q )                                          # Auxiliar variable
        # P_0_old = ( (det(M + 0.5*eye_2N))**(-0.5) )*np.exp( -1*np.matmul( Q.transpose() , np.matmul( np.linalg.pinv(2*M + eye_2N) , Q )  ) )                # Auxiliar variable
        # 
        # assert np.allclose(R  ,   R_old, rtol=1e-10, atol=1e-10), "Achei!"
        # assert np.allclose(y  ,   y_old, rtol=1e-10, atol=1e-10), "Achei!"
        # assert np.allclose(P_0, P_0_old, rtol=1e-10, atol=1e-10), "Achei!"
        # 
        # print("Passou")
        
        H = Hermite_multidimensional(R, y, n_cutoff)                            # Calculate the Hermite polynomial associated with this gaussian state
        
        # Calculate the probabilities
        P = np.zeros(self.N_modes*[n_cutoff])                                   # Initialize the tensor to 0 (n_cutoff entries in each of the self.N_modes dimensions)
        
        idx = np.ravel_multi_index((self.N_modes)*[0], dims=P.shape, order='F')            # Get "linearized" index
        P.ravel()[idx] = P_0                                                    # Set its first entry to P_0

        # Similar procedure to what precedes. Move forward in the P tensor and fill it element by element.
        nextP = np.ones([self.N_modes, 1], dtype=int);
        for jj in range(1, 1+n_cutoff**(self.N_modes)-1):            # jj = 1:n_cutoff^(dim/2) - 1
            
            
            for ii in range(1, 1+self.N_modes):                      #ii = 1:dim/2   # Figure out what the next coordinate to fill in is
                jumpTo = np.zeros([self.N_modes, 1], dtype=int);
                jumpTo[ii-1] = 1;
                
                if nextP[ii-1] + jumpTo[ii-1] > n_cutoff:
                   nextP[ii-1] = 1;
                else:
                   nextP[ii-1] = nextP[ii-1] + 1;
                   break
            
            nextCoord = np.ravel_multi_index(list(nextP-1), dims=P.shape, order='F')    # Get "linearized" index
            
            whichH = np.zeros([2*self.N_modes, 1], dtype=int);                  # Corresponding entry on Hermite polynomial
            whichH[:self.N_modes] = nextP                                    # Copy the position of the probability twice                 
            whichH[self.N_modes:] = nextP                                    # whichH = [nextP, nextP]
            
            # whichH = np.zeros([2*self.N_modes, 1], dtype=int);
            # for kk in range(self.N_modes):              # m = (n,n) -> Repeat the entries 
            #     whichH[kk] = nextP[kk];                 # m[0:N]   = n
            #     whichH[kk+self.N_modes] = nextP[kk];    # m[N:2*N] = n
            
            idx_H = np.ravel_multi_index(list(whichH-1), dims=H.shape, order='F') # Get "linearized" index whichH = num2cell(whichH);
            
            P.ravel()[nextCoord] = P_0*H.ravel()[idx_H];
            for kk in range(self.N_modes):                     # kk = 1:dim/2
                P.ravel()[int(nextCoord)] = P.ravel()[int(nextCoord)]/np.math.factorial(int(nextP[kk]-1));
        
        return P


################################################################################


def Hermite_multidimensional_original(R, y, n_cutoff=10):
    """
    Calculates the multidimensional Hermite polynomial H_m^R(y) from m = (0, ..., 0) up to (n_cutoff, ..., n_cutoff)
    
    ARGUMENTS:
        R - n by n symmetric matrix
        y - n-dimensional point where the polynomial is to be evaluated
        n_cutoff - maximum value for the polynomial to be calculated
        
    RETURNS:
        H - tensor with the multidimensional Hermite polynomial evaluated on n-dimensional grid ( H.shape = n*[n_cutoff] )
    
    REFERENCE:
        Math. Comp. 24, 537-545 (1970)
    """
    n = len(R)                                                                  # Dimension of the input matrix and vector
    
    H = np.zeros(n*[n_cutoff], dtype=complex)                                   # Initialize the tensor to 0 (n_cutoff entries in each of the dim dimensions)

    m_last = np.array(n*[0], dtype=int)                                         # Linear index for last altered entry of the Hermite tensor
    m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')       # Get "linearized" index (Adjust from Python indexig to original article indexing starting a 1)
    H.ravel()[m_last_linear] = 1                                                # Set its first entry to 1 (known value)
    
    n_entries = np.prod(H.shape)                                                # Number of entries on the final tensor
    
    for m_next_linear in range(1, n_entries):                                 # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
        m_next = np.array(np.unravel_index(m_next_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
        
        e_k = m_next - m_last                                                   # Discover how much it has transversed since last iteration
        
        if np.any(e_k<0):                                                       # If it changed the dimension transversed
            m_last[e_k<0] = 0                                                   # Move the last position accordingly
            m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F') # Update the last linear index accordingly
            
            e_k[e_k<0] = 0                                                      # Remember to alter notation (e_k must be only zeros and a single one)
        
        k = np.where(e_k.squeeze())[0]                                          # Index for the entry where e_k == 1 (should be only one entry that matches this condition!)
        
        # Calculate the first term of this new entry
        R_times_y = 0
        for j in range(n):                                                      # This loop is essentially the sum on this first term
            R_times_y = R_times_y + R[k,j]*y[j,0]
            
        H.ravel()[m_next_linear] = R_times_y*H.ravel()[m_last_linear]           # Remember that m_last = m_next - e_k
        
        #  Calculate the second term of this new entry
        for j in range(n):
            e_j = np.zeros(n, dtype=int)
            e_j[j] = 1                                                          # For this j, build the vector e_j
            
            m_jk = m_last - e_j
            if (j == k) or np.any(m_jk < 0):                                    # If you end up with a negative index 
                continue                                                        # the corresponding entry of the tensor is null
            
            m_jk_linear = np.ravel_multi_index(m_jk, dims=H.shape, order='F')
            H.ravel()[m_next_linear] = H.ravel()[m_next_linear] - m_next[j]*R[k,j]*H.ravel()[m_jk_linear]
            
        #  Calculate the last term of this new entry
        m_2k = m_next - 2*e_k
        if np.all(m_2k >= 0):
            m_2k_linear = np.ravel_multi_index(m_2k, dims=H.shape, order='F')
            H.ravel()[m_next_linear] =  H.ravel()[m_next_linear] - R[k,k]*(m_next[k]-1)*H.ravel()[m_2k_linear]
        
        # Update the last index before moving to the next iteration
        m_last = m_next                                                         # Update the last vector index of the loop
        m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')   # Update the last linear index of the loop

    H = H.real                                                                  # Get rid off any residual complex value
    
def Hermite_multidimensional(R, y, n_cutoff=10):
    """
    Calculates the multidimensional Hermite polynomial H_m^R(y) from m = (0, ..., 0) up to (n_cutoff, ..., n_cutoff)
    
    ARGUMENTS:
        R - n by n symmetric matrix
        y - n-dimensional point where the polynomial is to be evaluated
        n_cutoff - maximum value for the polynomial to be calculated
        
    RETURNS:
        H - tensor with the multidimensional Hermite polynomial evaluated on n-dimensional grid ( H.shape = n*[n_cutoff] )
    
    REFERENCE:
        Math. Comp. 24, 537-545 (1970)
    """
    n = len(R)                                                                  # Dimension of the input matrix and vector
    
    H = np.zeros(n*[n_cutoff], dtype=complex)                                   # Initialize the tensor to 0 (n_cutoff entries in each of the dim dimensions)

    m_last = np.array(n*[0], dtype=int)                                         # Linear index for last altered entry of the Hermite tensor
    m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')       # Get "linearized" index (Adjust from Python indexig to original article indexing starting a 1)
    H.ravel()[m_last_linear] = 1                                                # Set its first entry to 1 (known value)
    
    n_entries = np.prod(H.shape)                                                # Number of entries on the final tensor
    
    for m_next_linear in range(1, n_entries):                                 # Loop through every entry on tensor H using linear indices ( m is the linearized index for H: H.ravel()[m] <-> H[ np.unravel_index(m, H.shape, order='F') ] )
        
        m_next = np.array(np.unravel_index(m_next_linear, H.shape, order='F'), dtype=int)  # Vector index for the next entry of the Hermite tensor to be calculated
        
        e_k = m_next - m_last                                                   # Discover how much it has transversed since last iteration
        
        if np.any(e_k<0):                                                       # If it changed the dimension transversed
            m_last[e_k<0] = 0                                                   # Move the last position accordingly
            m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F') # Update the last linear index accordingly
            
            e_k[e_k<0] = 0                                                      # Remember to alter notation (e_k must be only zeros and a single one)
        
        k = np.where(e_k.squeeze())[0]                                          # Index for the entry where e_k == 1 (should be only one entry that matches this condition!)
        
        # Debugging
        # if np.any(m_last<0):
        #     a=1
        
        # Calculate the first term of this new entry
        R_times_y = 0
        for j in range(n):                                                      # This loop is essentially the sum on this first term
            R_times_y = R_times_y + R[k,j]*y[j,0]
            
        H.ravel()[m_next_linear] = R_times_y*H.ravel()[m_last_linear]           # Remember that m_last = m_next - e_k
        
        #  Calculate the second term of this new entry
        for j in range(n):
            e_j = np.zeros(n, dtype=int)
            e_j[j] = 1                                                          # For this j, build the vector e_j
            
            m_jk = m_last - e_j
            if (j==k) or np.any(m_jk < 0):                                                # If you end up with a negative index 
                continue                                                        # the corresponding entry of the tensor is null
            
            m_jk_linear = np.ravel_multi_index(m_jk, dims=H.shape, order='F')
            H.ravel()[m_next_linear] = H.ravel()[m_next_linear] - m_next[j]*R[k,j]*H.ravel()[m_jk_linear]
            
        #  Calculate the last term of this new entry
        m_2k = m_next - 2*e_k
        if np.all(m_2k >= 0):
            m_2k_linear = np.ravel_multi_index(m_2k, dims=H.shape, order='F')
            H.ravel()[m_next_linear] =  H.ravel()[m_next_linear] - R[k,k]*(m_next[k]-1)*H.ravel()[m_2k_linear]
        
        # Update the last index before moving to the next iteration
        m_last = m_next                                                         # Update the last vector index of the loop
        m_last_linear = np.ravel_multi_index(m_last, dims=H.shape, order='F')   # Update the last linear index of the loop

    H = H.real                                                                  # Get rid off any residual complex value
    
    return H

def is_a_function(maybe_a_function):
    """
    Auxiliar internal function checking if a given variable is a lambda function
    """
    return callable(maybe_a_function)                   # OLD: isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"

def lyapunov_ode_unconditional(t, V_old_vector, A, D):
    """
    Auxiliar internal function defining the Lyapunov equation 
    and calculating the derivative of the covariance matrix
    """
    
    M = A.shape[0];                                                             # System dimension (N_particles + 1 cavity field)partículas  + 1 campo)
    
    A_T = np.transpose(A)                                                       # Transpose of the drift matrix
    
    V_old = np.reshape(V_old_vector, (M, M));                                      # Vector -> matrix
    
    dVdt = np.matmul(A, V_old) + np.matmul(V_old, A_T) + D;                     # Calculate how much the CM derivative in this time step
    
    dVdt_vector = np.reshape(dVdt, (M**2,));                                     # Matrix -> vector
    return dVdt_vector

def lyapunov_ode_conditional(t, V_old_vector, A, D, B):
    """
    Auxiliar internal function defining the Lyapunov equation 
    and calculating the derivative of the covariance matrix
    """
    
    M = A.shape[0];                                                             # System dimension (N_particles + 1 cavity field)partículas  + 1 campo)
    
    A_T = np.transpose(A)                                                       # Transpose of the drift matrix
    
    V_old = np.reshape(V_old_vector, (M, M));                                   # Vector -> matrix
    
    # chi = np.matmul(C, V_old) + Gamma             # THIS SHOULD BE FASTER!
    # chi = np.matmul(np.transpose(chi), chi)
    # chi = np.matmul( np.matmul(V_old, np.transpose(C)) + np.transpose(Gamma),  np.matmul(C, V_old) + Gamma )    # Auxiliar matrix
    chi = np.matmul( np.matmul( np.matmul(V_old, B), np.transpose(B)), V_old)    # Auxiliar matrix
    
    dVdt = np.matmul(A, V_old) + np.matmul(V_old, A_T) + D - chi;               # Calculate how much the CM derivative in this time step
    
    dVdt_vector = np.reshape(dVdt, (M**2,));                                    # Matrix -> vector
    return dVdt_vector


################################################################################


class gaussian_dynamics:
    """Class simulating unconditional and conditional dynamics of a gaussian state following a set of Langevin and Lyapunov equations
    
    ATTRIBUTES
        A                     - Drift matrix (can be a callable lambda functions to have a time dependency!)
        D                     - Diffusion Matrix 
        N                     - Mean values of the noises
        initial_state         - Initial state of the global system
        t                     - Array with timestamps for the time evolution
        
        is_stable             - Boolean telling if the system is stable or not
        N_time                - Length of time array
        Size_matrices         - Size of covariance, diffusion and drift matrices
        
        states_unconditional  - List of time evolved states following unconditional dynamics
        states_conditional    - List of time evolved states following   conditional dynamics (mean quadratures are the average of the trajectories from the quantum monte carlo method)
        steady_state_internal - Steady state
        
        quantum_trajectories        - Quantum trajectories from the Monte Carlo method for the conditional dynamics
        semi_classical_trajectories - List of time evolved semi-classical mean quadratures (Semi-classical Monte Carlo method)
    """
    
    def __init__(self, A_0, D_0, N_0, initial_state_0):
        """Class constructor for simulating the time evolution of the multimode systems following open unconditional and conditional quantum dynamics dictated by Langevin and Lyapunov equations
        
        Langevin: \dot{R} = A*X + N           : time evolution of the mean quadratures
       
        Lyapunov: \dot{V} = A*V + V*A^T + D   : time evolution of the covariance matrix
       
        PARAMETERS:
           A_0           - Drift matrix     (numerical matrix or callable function for a time-dependent matrix)
           D_0           - Diffusion Matrix (auto correlation of the noises, assumed to be delta-correlated in time)
           N_0           - Mean values of the noises
           initial_state - Cavity linewidth
       
        BUILDS:
           self           - instance of a time_evolution class
           self.is_stable - boolean telling if the system is stable or not
        """
      
        self.A = A_0;  # .copy() ?                                              # Drift matrix
        self.D = D_0;  # .copy() ?                                              # Diffusion Matrix
        self.N = N_0.reshape((len(N_0),1));   # .copy() ?                       # Mean values of the noises
        
        self.initial_state = initial_state_0;                                   # Initial state of the global system
        
        self.Size_matrices = len(self.D);                                       # Size of system and ccupation number for the environment (heat bath)
      
        # assert 2*initial_state_0.N_modes == self.Size_matrices), "Initial state's number of modes does not match the drift and diffusion matrices sizes"              # Check if the initial state and diffusion/drift matrices have appropriate sizes !
      
        if( not is_a_function(self.A) ):
            eigvalue, eigvector = np.linalg.eig(self.A);                        # Calculate the eigenvalues of the drift matrix
            is_not_stable = np.any( eigvalue.real > 0 );                        # Check if any eigenvalue has positive real part (unstability)
            self.is_stable = not is_not_stable                                  # Store the information of the stability of the system in a class attribute
    
    def unconditional_dynamics(self, t_span):
        """Calculates the time evolution of the initial state following an unconditional dynamics at the input timestamps.
        
       PARAMETERS:
           tspan - Array with time stamps when the calculations should be done
       
       CALCULATES: 
           self.states_conditional - list of gaussian_state instances with the time evolved gaussian states for each timestamp of the input argument t_span
       
        RETURNS:
            result - list of gaussian_state instances with the time evolved gaussian states for each timestamp of the input argument t_span
        """
      
        R_evolved, status_langevin = self.langevin(t_span);                     # Calculate the mean quadratures for each timestamp
      
        V_evolved, status_lyapunov = self.lyapunov(t_span);                     # Calculate the CM for each timestamp (perform time integration of the Lyapunov equation)
        
        assert status_langevin != -1 and status_lyapunov != -1, "Unable to perform the time evolution of the unconditional dynamics - Integration step failed"         # Make sure the parameters for the time evolution are on the correct order of magnitude!
                
        self.states_unconditional = []                                          # Combine the time evolutions calculated above into an array of gaussian states
        for i in range(self.N_time):
            self.states_unconditional.append( gaussian_state(R_evolved[:, i], V_evolved[i]) );
        
        result = self.states_unconditional;
        return result                                                           # Return the array of time evolved gaussian_state following unconditional dynamics
    
    def langevin(self, t_span):
        """Solve quantum Langevin equations for the time evolved mean quadratures of the multimode systems
       
        Uses ode45 to numerically integrate the average Langevin equations (a fourth order Runge-Kutta method)
       
        PARAMETERS:
            t_span - timestamps when the time evolution is to be calculated (ndarray)
       
        CALCULATES:
            self.R - a cell with the time evolved mean quadratures where self.R(i,j) is the i-th mean quadrature at the j-th timestamp
        """
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            langevin_ode = lambda t, R: np.reshape(np.matmul(self.A(t), R.reshape((len(R),1))) + self.N, (len(R),))        # Function handle that defines the Langevin equation (returns the derivative)
        else:
            langevin_ode = lambda t, R: np.reshape(np.matmul(self.A, np.reshape(R, (len(R),1))) + self.N, (len(R),))           # Function handle that defines the Langevin equation (returns the derivative)
        
        solution_langevin = solve_ivp(langevin_ode, [t_span[0], t_span[-1]], np.reshape(self.initial_state.R, (self.Size_matrices,)), t_eval=t_span) # Solve Langevin eqaution through Runge Kutta(4,5)
        # Each row in R corresponds to the solution at the value returned in the corresponding row of self.t
        
        R_evolved = solution_langevin.y;                                        # Store the time evolved quadratures in a class attribute
        
        return R_evolved, solution_langevin.status
    
    def lyapunov(self, t_span, is_conditional=False, AA=0, DD=0, B=0):
        """Solve the lyapunov equation for the time evolved covariance matrix of the full system (both conditional and unconditional cases)
       
        Uses ode45 to numerically integrate the Lyapunov equation, a fourth order Runge-Kutta method
       
        PARAMETERS:
            t_span - timestamps when the time evolution is to be calculated
       
        CALCULATES:
            'self.V' - a cell with the time evolved covariance matrix where self.V[j] is the covariance matrix at the j-th timestamp in t_span
        """
        
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        V_0_vector = np.reshape(self.initial_state.V, (self.Size_matrices**2, )); # Reshape the initial condition into a vector (expected input for ode45)
        
        if is_conditional:                                                      # Check if the dynamics is conditional or unconditional
            if is_a_function(self.A):                                           # Check if there is a time_dependency on the odes
                ode = lambda t, V: lyapunov_ode_conditional(t, V, self.A(t)+AA, self.D+DD, B); # Function handle that defines the Langevin equation (returns the derivative)
            else:
                ode = lambda t, V: lyapunov_ode_conditional(t, V, self.A+AA   , self.D+DD, B);    # Lambda unction that defines the Lyapunov equation (returns the derivative)
        else:
            if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
                ode = lambda t, V: lyapunov_ode_unconditional(t, V, self.A(t), self.D); # Function handle that defines the Langevin equation (returns the derivative)
            else:
                ode = lambda t, V: lyapunov_ode_unconditional(t, V, self.A, self.D);    # Lambda unction that defines the Lyapunov equation (returns the derivative)
                
        solution_lyapunov = solve_ivp(ode, [t_span[0], t_span[-1]], V_0_vector, t_eval=t_span) # Solve Lyapunov equation through Fourth order Runge Kutta
        
        # Unpack the output of ode45 into a list where each entry contains the information about the evolved CM at each time
        V_evolved = []                                                          # Initialize a cell to store the time evolvd CMs for each time
        
        for i in range(len(solution_lyapunov.t)):
            V_current_vector = solution_lyapunov.y[:,i];                                        # Take the full Covariance matrix in vector form
            V_current = np.reshape(V_current_vector, (self.Size_matrices, self.Size_matrices)); # Reshape it into a proper matrix
            V_evolved.append(V_current);                                                        # Append it on the class attribute
                    
        return V_evolved, solution_lyapunov.status
        
    def steady_state(self, A_0=0, A_c=0, A_s=0, omega=0): # *args -> CONSERTAR !
        """Calculates the steady state for the multimode system
       
        PARAMETERS (only needed if the drift matrix has a periodic time dependency):
          A_0, A_c, A_s - Components of the Floquet decomposition of the drift matrix
          omega         - Frequency of the drift matrix
        
        CALCULATES:
            self.steady_state_internal - gaussian_state with steady state of the system
          
        RETURNS:
            ss - gaussian_state with steady state of the system
        """
      
        if is_a_function(self.A):                                               # If the Langevin and Lyapunov eqs. have a time dependency, move to the Floquet solution
            ss = self.floquet(A_0, A_c, A_s, omega);
            self.steady_state_internal = ss;                                    # Store it in the class instance
        
        else:                                                                   # If the above odes are time independent, 
            assert self.is_stable, "There is no steady state covariance matrix, as the system is not stable!"  # Check if there exist a steady state!
        
            R_ss = np.linalg.solve(self.A, -self.N);                            # Calculate steady-state mean quadratures
            V_ss = solve_continuous_lyapunov(self.A, -self.D);                  # Calculate steady-state covariance matrix
        
            ss = gaussian_state(R_ss, V_ss);                                    # Generate the steady state
            self.steady_state_internal = ss;                                    # Store it in the class instance
            
        return ss                                                               # Return the gaussian_state with the steady state for this system
    
    def floquet(self, A_0, A_c, A_s, omega):
        """Calculates the staeady state of a system with periodic Hamiltonin/drift matrix
        
        Uses first order approximation in Floquet space for this calculation
       
        Higher order approximations will be implemented in the future
        
        PARAMETERS:
          A_0, A_c, A_s - components of the Floquet decomposition of the drift matrix
          omega - Frequency of the drift matrix
        
        CALCULATES:
          self.steady_state_internal - gaussian_state with steady state of the system
          
        RETURNS:
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
        
        R_ss = R_ss_F[0:M];                                                     # Get only the first entries
        V_ss = V_ss_F[0:M, 0:M];                                                # Get only the first sub-matrix
        
        ss = gaussian_state(R_ss, V_ss);                                        # Generate the steady state
        self.steady_state_internal = ss;                                        # Store it in the class instance
        
        return ss
    
    def semi_classical(self, t_span, N_ensemble=2e+2):
        """Solve the semi-classical Langevin equation for the expectation value of the quadrature operators using a Monte Carlos simulation to numerically integrate the Langevin equations
        
        The initial conditions follows the initial state probability density in phase space
        The differential stochastic equations are solved through a Euler-Maruyama method
       
        PARAMETERS:
          t_span - timestamps when the time evolution is to be calculated
          N_ensemble (optional) - number of iterations for Monte Carlos simulation, default value: 200
       
        CALCULATES:
          self.R_semi_classical - matrix with the quadratures expectation values of the time evolved system where 
          self.R_semi_classical(i,j) is the i-th quadrature expectation value at the j-th time
        """
      
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        dt = self.t[2] - self.t[1];                                             # Time step
        sq_dt =  np.sqrt(dt);                                                   # Square root of time step (for Wiener proccess in the stochastic integration)
        
        noise_amplitude = self.N + np.sqrt( np.diag(self.D) );                  # Amplitude for the noises (square root of the auto correlations)
        
        mean_0 = self.initial_state.R;                                          # Initial mean value
        std_deviation_0 =  np.sqrt( np.diag(self.initial_state.V) );            # Initial standard deviation
        
        self.semi_classical_trajectories = np.zeros((self.Size_matrices, self.N_time));    # Matrix to store each quadrature ensemble average at each time
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            AA = lambda t: self.A(t);                                           # Rename the function that calculates the drift matrix at each time
        else:
            AA = lambda t: self.A;                                              # If A is does not vary in time, the new function always returns the same value 
      
        for i in range(N_ensemble):                                             # Loop on the random initial positions (# Monte Carlos simulation using Euler-Maruyama method in each iteration)
            
            X = np.zeros((self.Size_matrices, self.N_time));                    # For this iteration, this matrix stores each quadrature at each time (first and second dimensions, respectively)
            X[:,0] = np.random.normal(mean_0, std_deviation_0)                  # Initial Cavity position quadrature (normal distribution for vacuum state)
            
            noise = np.random.standard_normal(X.shape);
            for k in range(self.N_time-1):                                      # Euler-Maruyama method for stochastic integration
                X[:,k+1] = X[:,k] + (np.matmul(AA(self.t[k]), X[:,k]) + self.N)*dt + sq_dt*np.multiply(noise_amplitude, noise[:,k])
                                   
            self.semi_classical_trajectories = self.semi_classical_trajectories + X;    # Add the new  Monte Carlos iteration quadratures to the same matrix
        
        self.semi_classical_trajectories = self.semi_classical_trajectories/N_ensemble; # Divide the ensemble sum to obtain the average quadratures at each time
        
        result = self.semi_classical_trajectories
        return result
        
    def langevin_conditional(self, t_span, V_evolved, N_ensemble=200, rho_bath=gaussian_state(), C=0, Gamma=0, V_m=0):
        """Solve the conditional stochastic Langevin equation for the expectation value of the quadrature operators
        using a Monte Carlos simulation to numericaly integrate the stochastic Langevin equations
        
        The differential stochastic equations are solved through a Euler-Maruyama method
       
        PARAMETERS:
          t_span     - timestamps when the time evolution is to be calculated
          N_ensemble - number of iterations for Monte Carlos simulation, default value: 200
          rho_bath   - gaussian_state with the quantum state of the environment's state
          C          - matrix describing the measurement process (see conditional_dynamics)
          Gamma      - matrix describing the measurement process (see conditional_dynamics)
          V_m        - Covariance matrix of the post measurement state
       
        CALCULATES:
          self.quantum_trajectories - list of single realizations of the quantum Monte Carlo method for the mean quadrature vector
        
        RETURN:
          R_conditional - average over the trajectories of the quadrature expectation values
        """
        
        N_ensemble = int(N_ensemble)
        
        self.t = t_span;                                                        # Timestamps for the simulation
        self.N_time = len(t_span);                                              # Number of timestamps
        
        dt = self.t[2] - self.t[1];                                             # Time step
        sq_dt_2 =  np.sqrt(dt)/2.0;                                             # Square root of time step (for Wiener proccess in the stochastic integration)
        
        R_conditional = np.zeros((self.Size_matrices, self.N_time));            # Matrix to store each quadrature ensemble average at each time
        
        if is_a_function(self.A):                                               # I have to check if there is a time_dependency on the odes
            AA = lambda t: self.A(t);                                           # Rename the function that calculates the drift matrix at each time
        else:
            AA = lambda t: self.A;                                              # If A is does not vary in time, the new function always returns the same value 
        
        N_measured2 = C.shape[0]                                                 # Number of modes to be measured
        C_T = np.transpose(C)
        Gamma_T = np.transpose(Gamma)
        
        R_bath = np.reshape(rho_bath.R, (len(rho_bath.R),))                                   # Mean quadratures for the bath state
        V_bath = rho_bath.V
        
        N = np.reshape(self.N, (len(self.N),))
        
        self.quantum_trajectories = N_ensemble*[None]                           # Preallocate memory to store the trajectories
        
        for i in range(N_ensemble):                                             # Loop on the random initial positions (# Monte Carlos simulation using Euler-Maruyama method in each iteration)
            
            X = np.zeros((self.Size_matrices, self.N_time));                    # Current quatum trajectory to be calculated, this matrix stores each quadrature at each time (first and second dimensions, respectively)
            X[:,0] = np.reshape(self.initial_state.R, (2*self.initial_state.N_modes,))                     # Initial mean quadratures are exactly the same as the initial state (stochasticity only appear on the measurement outcomes)
            
            cov = (V_bath + V_m)/2.0                                            # Covariance for the distribution of the measurement outcome (R_m)
            R_m = np.random.multivariate_normal(R_bath, cov, (self.N_time))     # Sort the measurement results
            
            for k in range(self.N_time-1):                                      # Euler-Maruyama method for stochastic integration of the conditional Langevin equation
                V = V_evolved[k]                                       # Get current covariance matrix (pre-calculated according to deterministic conditional Lyapunov equation)
                dw = np.matmul(fractional_matrix_power(V_bath+V_m, -0.5), R_m[k,:] - R_bath) # Calculate the Wiener increment
                
                X[:,k+1] = X[:,k] + (np.matmul(AA(self.t[k]), X[:,k]) + N)*dt + sq_dt_2*np.matmul(np.matmul(V, C_T) + Gamma_T, dw) # Calculate the quantum trajectories following the stochastis conditional Langevin equation
            
            self.quantum_trajectories[i] = X                                    # Store each trajectory into class instance                
            
            R_conditional = R_conditional + X;                                  # Add the new quantum trajectory in order to calculate the average
        
        R_conditional = R_conditional/N_ensemble;                               # Divide the ensemble sum to obtain the average quadratures at each time
        
        return R_conditional
    
    def conditional_dynamics(self, t_span, N_ensemble=1e+2, C_int=None, rho_bath=None, s_list = [1], phi_list=None):
        """Calculates the time evolution of the initial state following a conditional dynamics at the input timestamps
        
        Independent general-dyne measurements can be applied to each mode of the multimode gassian state with N modes
        
        PARAMETERS:
            tspan    - numpy.ndarray with time stamps when the calculations should be done
            C_int    - interaction matrix between system and bath
            rho_bath - gaussian state of the bath
            s        - list of measurement parameters for each measured mode (s=1: Heterodyne ; s=0: Homodyne in x-quadrature ; s=Inf: Homodyne in p-quadrature)
            phi      - list of directions on phase space of the measurement for each measured mode
            
        CALCULATES:
            self.states_conditional - list of time evolved gaussian_state for each timestamp of the input argument t_span
            
        RETURNS:
            result - list of time evolved gaussian_state for each timestamp of the input argument t_span
        """
        
        # TODO: generalize for arbitrary number of monitored modes        
        # TODO: Independent measurements can be applied to the last k modes of the multimode gassian state with N modes
        # N_measured = len(s_list)                                              # Number of monitored modes
        # Omega_m = rho_bath.Omega                                              # Symplectic form matrix for the monitored modes
        # Omega_n = self.initial_state.Omega                                    # Symplectic form matrix for the whole system
        # C = np.matmul(np.matmul(temp, Omega_m), C_int_T);                     # Extra matrix on the Lyapunov equation
        # Gamma = -np.matmul(np.matmul(np.matmul(temp, V_bath),C_int_T),Omega_n)# Extra matrix on the Lyapunov equation
        
        N_measured = self.initial_state.N_modes                                 # Number of monitored modes (currently all modes)
        Omega = self.initial_state.Omega                                        # Symplectic form matrix for the whole system
        
        if phi_list is None: phi_list = N_measured*[0]                          # If no measurement direction was indicated, use default value of 0 to all measured modes
        
        if C_int is None: C_int = np.eye(2*N_measured)                          # If no system-bath interaction bath was indicated, use as default value an identity matrix
        
        if rho_bath is None: rho_bath = vacuum(N_measured)                      # If no state for the environment was indicated, use  as default value a tensor product of vacuum states
        
        assert N_measured == len(phi_list), "conditional_dynamics can not infer the number of modes to be measured, number of measurement parameters is different from rotation angles"
        
        assert rho_bath.N_modes == N_measured, "The number of bath modes does not match the number of monitored modes"
        
        assert N_measured <= self.initial_state.N_modes, "There are more monitored modes, than there are on the initial state"
        
        V_bath = rho_bath.V                                                     # Covariance matrix for the bath's state
        
        V_m = 42                                                                # Just to initialize the variable, this value will be removed after the loop
        for i in range(N_measured):                                             # For each measurement parameter
            s = s_list[i]
            temp = np.block([[s, 0],[0, 1/s]])                                  # Calculate the associated mode's covariance matrix after measurement
            
            phi = phi_list[i]                                                   # Get rotation of measurement angle
            Rot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
            Rot_T = np.transpose(Rot)
            
            temp = np.matmul( np.matmul(Rot, temp), Rot_T)                      # Rotate measured covariance matrix
            
            V_m = block_diag(temp, V_m)                                         # Build the covariance matrix of the tensor product of these modes
        V_m = V_m[0:len(V_m)-1, 0:len(V_m)-1]                                   # Remove the initialization value
        
        temp = np.linalg.inv(V_bath + V_m)                     # Auxiliar variable
        temp_minus = fractional_matrix_power(V_bath + V_m, -0.5)
        
        B = np.matmul(np.matmul(C_int, Omega), temp_minus)                        # Extra matrix on the Lyapunov equation
        
        AA = - np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp), Omega), np.transpose(C_int))
        DD = + np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp), V_bath), np.transpose(C_int)), Omega)
        
        is_conditional = True                                                   # Boolean telling auxiliar variable that the conditional dynamics is to be calculated
        
        V_evolved, status_lyapunov = self.lyapunov(t_span, is_conditional, AA, DD, B);       # Calculate the deterministic dynamics for the CM for each timestamp (perform time integration of the Lyapunov equation)
        
        assert status_lyapunov != -1, "Unable to perform the time evolution of the covariance matrix through Lyapunov equation - Integration step failed"
        
        ################################################################################################################################
        C = np.transpose(-np.matmul(np.matmul(C_int, Omega), temp))              # Extra matrix on the Lyapunov equation
        
        Gamma = np.transpose(np.matmul(np.matmul(np.matmul(Omega, C_int), V_bath), temp)) # Extra matrix on the Lyapunov equation
        ################################################################################################################################
        
        R_evolved = self.langevin_conditional(t_span, V_evolved, N_ensemble, rho_bath, C, Gamma, V_m)  # Calculate the quantum trajectories and its average
        
        self.states_conditional = []                                            # Combine the time evolutions calculated above into an array of gaussian states
        for i in range(self.N_time):
            self.states_conditional.append( gaussian_state(R_evolved[:, i], V_evolved[i]) );        
      
        result = self.states_conditional;
        return result                                                           # Return the array of time evolved gaussian_state


################################################################################

# Create elementary gaussian states
def vacuum(N=1):
    """Returns an N-mode tensor product of vacuum states. Default N=1"""
    
    R = np.zeros(2*N)
    V = np.eye(2*N)
    
    return gaussian_state(R, V)

def coherent(alpha=1):
    """Returns a coherent state with complex amplitude alpha"""
    R = np.array([[2*alpha.real], [2*alpha.imag]]);                             # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.identity(2);                                                         # Covariance matrix of a coherent state with complex amplitude alpha
    
    return gaussian_state(R, V)

def squeezed(r=1):
    """Returns a squeezed state with real squeezing parameter r"""
    assert np.isreal(r), "Unsupported imaginary amplitude for squeezed state"
    
    R = np.array([[0], [0]])                                                    # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.diag([np.exp(-2*r), np.exp(+2*r)]);                                  # Covariance matrix of a coherent state with complex amplitude alpha
    
    return gaussian_state(R, V)

def thermal(nbar=1):
    """Returns a thermal state with mean occupation number nbar"""
    assert nbar>=0, "Imaginary or negative occupation number for thermal state"
    
    R = np.array([[0], [0]])                                                    # Mean quadratures  of a coherent state with complex amplitude alpha
    V = np.diag([2.0*nbar+1, 2.0*nbar+1]);                                      # Covariance matrix of a coherent state with complex amplitude alpha  
    
    return gaussian_state(R, V)


# Construct another state, from a base gaussian_state
def tensor_product(rho_list):
    state_copy = rho_list[0].copy()
    state_copy.tensor_product(rho_list[1:])
    
    return state_copy

def partial_trace(state, indexes):
    state_copy = state.copy()
    state_copy.partial_trace(indexes)
    
    return state_copy

def only_modes(state, indexes):
    state_copy = state.copy()
    state_copy.only_modes(indexes)
    
    return state_copy

def check_uncertainty_relation(state):
    return state.check_uncertainty_relation()

def loss_ancilla(state,idx,tau):
    state_copy = state.copy()
    state_copy.loss_ancilla(state,idx,tau)
    
    return state_copy


# Properties of a gaussian state
def symplectic_eigenvalues(state):
    return state.symplectic_eigenvalues()

def purity(state):
    return state.purity()

def squeezing_degree(state):
    return state.squeezing_degree()

def von_Neumann_Entropy(state):
    return state.von_Neumann_Entropy()

def mutual_information(state):
    return state.mutual_information()

def occupation_number(state):
    return state.occupation_number()

def number_operator_moments(state):
    return state.number_operator_moments()

def coherence(state):
    return state.coherence()

def logarithmic_negativity(state, *args):
    return state.logarithmic_negativity(*args)

def fidelity(rho_1, rho_2):
    return rho_1.fidelity(rho_2)

# Density matrix elements
def density_matrix_coherent_basis(state, alpha, beta):
    return state.coherence(alpha, beta)

def density_matrix_number_basis(state, n_cutoff=10):
    return state.density_matrix_number_basis(n_cutoff)

def number_statistics(state, n_cutoff=10):
    return state.number_statistics(n_cutoff)


# Gaussian unitaries (applicable to single mode states)
def displace(state, alpha, modes=[0]):
    state_copy = state.copy()
    state_copy.displace(alpha, modes)
    
    return state_copy

def squeeze(state, r, modes=[0]):
    state_copy = state.copy()
    state_copy.squeeze(r, modes)
    
    return state_copy

def rotate(state, theta, modes=[0]):
    state_copy = state.copy()
    state_copy.rotate(theta, modes)
    
    return state_copy 

def phase(state, theta, modes=[0]):
    state_copy = state.copy()
    state_copy.phase(theta, modes)
    
    return state_copy

# Gaussian unitaries (applicable to two mode states)
def beam_splitter(state, tau, modes=[0, 1]):
    state_copy = state.copy()
    state_copy.beam_splitter(tau, modes)
    
    return state_copy

def two_mode_squeezing(state, r, modes=[0, 1]):
    state_copy = state.copy()
    state_copy.two_mode_squeezing(r, modes)
    
    return state_copy

# Generic multimode gaussian unitary
def apply_unitary(state, S, d):
    state_copy = state.copy()
    state_copy.apply_unitary(S, d)
    
    return state_copy


# Gaussian measurements
def measurement_general(state, *args):
    state_copy = state.copy()
    state_copy.measurement_general(*args)
    
    return state_copy

def measurement_homodyne(state, *args):
    state_copy = state.copy()
    state_copy.measurement_homodyne(*args)
    
    return state_copy

def measurement_heterodyne(state, *args):
    state_copy = state.copy()
    state_copy.measurement_heterodyne(*args)
    
    return state_copy


