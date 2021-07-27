# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:57:16 2021

@author: igorb
"""
import numpy as np
from scipy.linalg import block_diag

class gaussian_state:                                                           # Class definning a multimode gaussian state
    # Constructor and its auxiliar functions    
    def __init__(self, *args):
        """
        Class simulating a gaussian state with mean quadratures and covariance matrix
        
        The user can explicitly pass the first two moments of a multimode gaussian state
        or pass a name-value pair argument to choose a single mode gaussian state
        
        ATRIBUTES:
        self.R       - Mean quadratures
        self.V       - Covariance matrix
        self.Omega   - Symplectic form matrix
        self.N_modes - Number of modes
        
        PARAMETERS:
          R0 - mean quadratures for gaussian state
          V0 - covariance matrix for gaussian state
          Alternatively, the user may pass a name-value pair argument 
          to create an elementary single mode gaussian state, see below.
        
        NAME-VALUE PAIR ARGUMENTS:
          "vacuum"                        - generates vacuum   state
          "thermal" , occupation number   - generates thermal  state
          "coherent", complex amplitude   - generates coherent state
          "squeezed", squeezing parameter - generates squeezed state
        """

        if(len(args) == 0):                                                     # Default constructor (vacuum state)
            self.R = np.array([[0], [0]])                                       # Save mean quadratres   in a class attribute
            self.V = np.identity(2)                                             # Save covariance matrix in a class attribute
            self.N_modes = 1;
             
        elif( isinstance(args[0], str) ):                                       # If the user called for an elementary gaussian state
            self.decide_which_state(args)                                       # Call the proper method to decipher which state the user wants 
        
        elif(isinstance(args[0], np.ndarray) & isinstance(args[1], np.ndarray)): # If the user gave the desired mean quadratures values and covariance matrix
            R0 = args[0];
            V0 = args[1];
            
            R_is_real = all(np.isreal(R0))
            R_is_vector = np.squeeze(R0).ndim == 1
            
            V_is_matrix = np.squeeze(V0).ndim == 2
            V_is_square = V0.shape[0] == V0.shape[1]
            
            R_and_V_match = len(R0) == len(V0)
            
            assert R_is_real & R_is_vector & V_is_matrix & R_and_V_match & V_is_square, "Unexpected first moments when creating gaussian state!"  # Make sure they are a vector and a matrix with same length
        
            self.R = np.vstack(R0);                                             # Save mean quadratres   in a class attribute (vstack to ensure column vector)
            self.V = V0;                                                        # Save covariance matrix in a class attribute
            self.N_modes = int(len(R0)/2);                                           # Save the number of modes of the multimode state in a class attribute
            
        else:
            raise ValueError('Unexpected arguments when creating gaussian state!') # If input arguments do not make sense, call out the user
        
        omega = np.array([[0, 1], [-1, 0]]);                                    # Auxiliar variable
        self.Omega = np.kron(np.eye(self.N_modes,dtype=int), omega)             # Save the symplectic form matrix in a class attribute                                                    
    
    def check_uncertainty_relation(self):
      """
      Check if the generated covariance matrix indeed satisfies the uncertainty principle (debbugging)
      """
      
      V_check = self.V + 1j*self.Omega;
      eigvalue, eigvector = np.linalg.eig(V_check)
      
      assert all(eigvalue>=0), "CM does not satisfy uncertainty relation!"
      
      return V_check
    
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
      
        rho = gaussian_state(R_final, V_final);                                 # Generate the gaussian state with these moments
      
        return rho
    
    def partial_trace(self, indexes):
        """
        Partial trace over specific single modes of the complete gaussian state
        
        PARAMETERS:
           indexes - the modes the user wants to trace out (as in the mathematical notation) 
        
        CALCULATES:
           rho_A - gaussian_state with all of the input state, except of the modes specified in 'indexes'
        """
      
        N_A = int(len(self.R) - 2*len(indexes));                                    # Twice the number of modes in resulting state
        assert N_A>=0, "Partial trace over more states than exists in gaussian state" 
      
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
      
        rho_A = gaussian_state(R0, V0)
        return rho_A
    
    def only_modes(self, indexes):
      """
      Partial trace over all modes except the ones in indexes of the complete gaussian state
       
       PARAMETERS:
          indexes - the modes the user wants to retrieve from the multimode gaussian state
      
       CALCULATES:
          rho - gaussian_state with all of the specified modes
      """
      
      N_A = len(indexes);                                                       # Twice the number of modes in resulting state
      assert N_A>0 & N_A <= self.N_modes, "Partial trace over more states than exists in gaussian state"
      
      R0 = np.zeros((2*N_A, 1))
      V0 = np.zeros((2*N_A, 2*N_A))
      
      for i in range(len(indexes)):
            m = indexes[i]
            R0[(2*i):(2*i+2)] = self.R[(2*m):(2*m+2)]
        
            for j in range(len(indexes)):
                n = indexes[j]
                V0[(2*i):(2*i+2), (2*j):(2*j+2)] = self.V[(2*m):(2*m+2), (2*n):(2*n+2)]
      
      rho_A = gaussian_state(R0, V0);
      return rho_A

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
            V_sq  - variance of the     squeezed quadrature
            V_asq - variance of the antisqueezed quadrature
            eta   - ratio of the variances above
       
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
        
        nu[nu==1] = nu[nu==1] + 5e-16;                                          # 0*log(0) is NaN, but in the limit that x->0 : x*log(x) -> 0
                                                                                # Doubles uses a 15 digits precision, I'm adding a noise at the limit of the numerical precision
                                          
        nu_plus  = (nu + 1)/2.0;                                                # Temporary variables
        nu_minus = (nu - 1)/2.0;
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
            single_mode = self.only_modes([j]);                                   # Get the covariance matrix for only the i-th mode
            S[j] = single_mode.von_Neumann_Entropy();                           # von Neumann Entropy for i-th mode of each covariance matrix
      
        S_tot = self.von_Neumann_Entropy();                                      # von Neumann Entropy for the total system of each covariance matrix
      
        I = np.sum(S) - S_tot;                                                     # Calculation of the mutual information
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
    
    # Gaussian unitaries
    # Applicable to single mode states
    def displace(self, alpha):
        """
        Apply displacement operator on a single mode gaussian state
        TO DO: generalize these operation to many modes!
       
        ARGUMENT:
           alpha - ampllitude for the displacement operator
        """
        
        assert self.N_modes   == 1, "Can only apply displacement operator on single mode state"
        d = np.array([[alpha.real], [alpha.imag]]);
        self.R = obj.R + d;
      
        # If a displacement is attempted at a whole array of states, it is possible to apply a displacement in every entry
        # however, I cannot see why this would be the desired effect, I prefer to consider an error
        # assert(all([obj.N_modes]) == 1, "Can only apply displacement operator on single mode state")
        
    def squeeze(self, r):
        """
        Apply squeezing operator on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           r - ampllitude for the squeezing operator
        """
        
        assert self.N_modes == 1, "Error with input arguments when trying to apply displacement operator"
        S = np.diag([np.exp(-r), np.exp(+r)]);
        
        self.R = np.matmul(S, self.R);
        self.V = np.matmul( np.matmul(S,self.V), S);
    
    
    def rotate(self, theta):
        """
        Apply phase rotation on a single mode gaussian state
        TO DO: generalize these operation to many modes!
        
        ARGUMENT:
           theta - ampllitude for the rotation operator
        """
        
        assert self.N_modes == 1, "Error with input arguments when trying to apply displacement operator"
        Rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        Rot_T = np.transpose(Rot)
        
        self.R = np.matmul(Rot, self.R);
        self.V = np.matmul( np.matmul(Rot, self.V), Rot_T);
    
    # Two mode states
    def beam_splitter(self, tau):
        """
        Apply a beam splitter transformation in a gaussian state
        tau - transmissivity of the beam splitter
        
        ARGUMENT:
           tau - ampllitude for the beam splitter operator
        """
        
        assert self.N_modes==2, "Beam splitter transformation can only be applied for a two mode system"
        
        B = np.sqrt(tau)*np.indentity(2)
        S = np.sqrt(1-tau)*np.indentity(2)
        BS = np.array([[B, S], [-S, B]])
        BS_T = np.transpose(BS)
        
        self.R = np.matmul(BS, self.R);
        self.V = np.matmul( np.matmul(BS, self.V), BS_T);
    
    def two_mode_squeezing(self, r):
        """
        Apply a two mode squeezing operator  in a gaussian state
        r - squeezing parameter
        
        ARGUMENT:
           r - ampllitude for the two-mode squeezing operator
        """
        
        assert self.N_modes==2, "Two mode squeezing operator can only be applied for a two mode system"
        
        S0 = np.cosh(r)*np.indentity(2);
        S1 = np.sinh(r)*np.diag([+1,-1]);
        S2 = np.array([[S0, S1], [S1, S0]])
        S2_T = np.transpose(S2)
        
        self.R = np.matmul(S2, self.R);
        self.V = np.matmul( np.matmul(S2, self.V, S2_T)
    
     
    
    
###############################################################################

a = gaussian_state("thermal", 100)

b = gaussian_state("squeezed", 1.2)
b.displace(2 + 5j)
b.tensor_product(b)
bb = b.tensor_product(b)
bb.two_mode_squeezing0.5)
bb.two_mode_squeezing(0.5)
b = bb.partial_trace([2])

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
