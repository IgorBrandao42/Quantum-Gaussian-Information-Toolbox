# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:57:16 2021

@author: igorb
"""
import numpy as np

class gaussian_state:                                                           # Class definning a multimode gaussian state
    def __init__(self, *args):
        # Class simulating a gaussian state with mean quadratures and covariance matrix
        # The user can explicitly pass the first two moments of a multimode gaussian state
        # or pass a name-value pair argument to choose a single mode gaussian state
        #
        # ATRIBUTES:
        # self.R       - Mean quadratures
        # self.V       - Covariance matrix
        # self.Omega   - Symplectic form matrix
        # self.N_modes - Number of modes
        #
        # PARAMETERS:
        #   R0 - mean quadratures for gaussian state
        #   V0 - covariance matrix for gaussian state
        #   Alternatively, the user may pass a name-value pair argument 
        #   to create an elementary single mode gaussian state, see below.
        # 
        # NAME-VALUE PAIR ARGUMENTS:
        #   "vacuum"                        - generates vacuum   state
        #   "thermal" , occupation number   - generates thermal  state
        #   "coherent", complex amplitude   - generates coherent state
        #   "squeezed", squeezing parameter - generates squeezed state

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
        
        # for i in range(self.N_modes):                                           # Build the symplectic form
        #     Omega = blkdiag(Omega, omega);
        # self.Omega = Omega;                                                     
    
    def check_uncertainty_relation(self):
      # Check if the generated covariance matrix indeed satisfies the uncertainty principle (debbugging)
      
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




R = np.array([1,2])
V = np.array([[10, 20],[30,40]])

a = gaussian_state()

b = gaussian_state(R, V)

c = gaussian_state("vacuum")







