'''
This tutorial covers how to construct a vector space in rom-tools
(https://pressio.github.io/rom-tools-and-workflows/romtools/vector_space.html)
'''

#First, we will import some relevant modules
import romtools
import numpy as np
from romtools import vector_space



if __name__ == "__main__":

    # Load snapshots from a FOM. Here, we use pre-computed snapshots of the 1D Euler equations
    # obtained using pressio-demo-apps
    snapshots = np.load('snapshots.npz')['snapshots']

    ## The snapshots are in tensor form:
    n_vars, nx, nt = snapshots.shape


    '''
    Now, let's construct a vector space using the snapshots. As a first example, we will construct 
    a vector space that simply uses the snapshots as a basis
    '''
  
    my_full_vector_space = vector_space.DictionaryVectorSpace(snapshots) 
   
    '''
    This returns an instance of the vector space class where the basis is
    equivalent to the snapshots and there is no affine offset
    '''
    assert np.allclose(my_full_vector_space.get_basis(), snapshots)
    assert np.allclose(my_full_vector_space.get_shift_vector(), 0.)


    '''
    Now we will do something more complicated by constructing a vector space
    with an affine offset corresponding to the initial conditions
    '''

    # First, create a shifter to use
    my_shifter = vector_space.create_firstvec_shifter(snapshots)
  
    # Now, create a   
