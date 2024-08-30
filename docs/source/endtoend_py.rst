Step-by-step in Python
======================

This step-by-step tutorial goes over how to use the PRESSIO ROM tools and workflow scripts on the
`2D reaction diffusion PDE problem <https://pressio.github.io/pressio-demoapps/diffusion_reaction_2d.html>`_ from pressio-demoapps.
The full example file is located at `https://github.com/Pressio/rom-tools-and-workflows <https://github.com/Pressio/rom-tools-and-workflows>`_.

Module Imports
***************
The following are the Python modules that are needed to be loaded for this example. ::

    # Pressio Modules
    import pressiodemoapps as pda
    import romtools as rt

    # Python Modules
    import os
    import math
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy

    # Jupyter Modules
    import ipywidgets as ipyw



Mesh Generation
***************

Below is an example of generating the mesh for a range of x and y resolutions.

The tool to generate the mesh is built into PRESSIO Demo Apps. ::

    def generate_mesh(pressio_file_path, mesh_path, figure_path, n_x, n_y):
        import os

        # generate mesh
        os.system('python3 ' + pressio_file_path + '/meshing_scripts/create_full_mesh_for.py --problem diffreac2d -n ' 
                    + str(n_x) + ' ' + str(n_y) + ' --outdir ' + mesh_path);

        # load mesh
        mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        # plot mesh
        x = mesh_obj.viewX()
        y = mesh_obj.viewY()
        unique_x = list(set(x))
        unique_y = list(set(y))
        ax.cla()
        for i in range(0,len(unique_x)):
            ax.vlines(unique_x[i], np.max(unique_y), np.min(unique_y), colors = 'k', linewidth = 0.5)
        for i in range(0,len(unique_y)):
            ax.hlines(unique_y[i], np.min(unique_x), np.max(unique_x), colors = 'k', linewidth = 0.5)

        return mesh_obj

    # Define paths
    mesh_path = 'mesh'
    figure_path = 'figs'

    # Generate and plot mesh
    fig = plt.figure(figsize=(10,7))
    fig.set_facecolor('w')
    ax = plt.gca()
    mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=25, n_y=25)
    plt.show()

This figure shows an example of the mesh for n_x = 50 and n_y = 50.

Full-Order Model
***************
This section will go over how to setup and run the full-order model (FOM).

First, we define our FOM Model class. It is important to note that this class follows the QoiModel class, which is required for greedy sampling. ::

    class FOM_Model(rt.workflows.models.QoiModel):
        def __init__(self):
            return None

        def populate_run_directory(self, run_directory, parameter_sample):
            os.system('ln -s mesh ' + run_directory + '/.')
            return 0

        def run_model(self, run_directory, parameter_sample):
            # Swap to run directory
            cdir = os.getcwd()
            os.chdir(run_directory)

            # Load mesh
            mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

            # Define Scheme
            # A. set scheme
            scheme  = pda.ViscousFluxReconstruction.FirstOrder

            # B. constructor for problem using default values
            prob_ID  = pda.DiffusionReaction2d.ProblemA
            self._problem = pda.create_problem(mesh_obj, prob_ID, scheme)

            # C. setting custom coefficients and custom source function
            self._problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, parameter_sample['D'], parameter_sample['K'])
            
            # (For Steady ROM)
            # D. Define residual function
            F = self._problem.createRightHandSide()
            def residual(x):
                self._problem.rightHandSide(x,0.,F)
                return F
            
            # Solve FOM
            yn = scipy.optimize.newton_krylov(residual, self._problem.initialCondition(), verbose=False)

            # Save solution
            np.savez('results.npz', y=yn, parameters=[parameter_sample['D'], parameter_sample['K']])

            # Swap back to base directory
            os.chdir(cdir)

            return 0 # NOTE: This function must return 0

        def compute_qoi(self, run_directory, parameter_sample):
            return np.load(run_directory + '/results.npz')['y']

    # Define FOM
    fom_model = FOM_Model()

Reduced-Order Model
***************
Here we define our reduced-order model (ROM) class. This follows the QoiModelWithErrorEstimate class. This specific model class is required for using greedy sampling for the ROM.

Of note in this model definition is an included check for whether we are using hyperreduction with the ROM, which changes how our sample indices, test basis, and approximation matrix are defined. ::

    class ROM_Model(rt.workflows.models.QoiModelWithErrorEstimate):
        def __init__(self, hyperreduction=False, offline_data_dir=''):
            # Load mesh
            self._mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

            # Load basis
            self._basis = np.load(offline_data_dir + '/basis.npz')['basis']

            # Hyperreduction
            self._hyperreduction = hyperreduction
            if self._hyperreduction:
                self._sample_indices = rt.hyper_reduction.deim_get_indices(self._basis)
                self._test_basis = rt.hyper_reduction.deim_get_test_basis(self._basis, self._basis, self._sample_indices)
                self._approx_mat = rt.hyper_reduction.deim_get_approximation_matrix(self._basis, self._sample_indices)
                np.savez('hyperreduction.npz', sample_indices=self._sample_indices, test_basis=self._test_basis, approx_mat=self._approx_mat)
            else:
                self._sample_indices = range(0,np.shape(self._basis)[0])
                self._test_basis = self._basis
                self._approx_mat = np.eye(np.shape(self._basis)[0])
            return None

        def populate_run_directory(self, run_directory, parameter_sample):
            # NOTE: This is needed when using myRK4
            os.system('ln -s mesh ' + run_directory + '/.')
            return 0

        def run_model(self, run_directory, parameter_sample):
            # Swap to run directory
            cdir = os.getcwd()
            os.chdir(run_directory)

            # Define Scheme
            # A. set scheme
            scheme  = pda.ViscousFluxReconstruction.FirstOrder

            # B. constructor for problem using default values
            prob_ID  = pda.DiffusionReaction2d.ProblemA
            problem = pda.create_problem(self._mesh_obj, prob_ID, scheme)

            # C. setting custom coefficients and custom source function
            problem = pda.create_diffusion_reaction_2d_problem_A(self._mesh_obj, scheme, my_source, parameter_sample['D'], parameter_sample['K'])

            # Run ROM
            # A. get initial condition
            yn = problem.initialCondition()
            qn = np.matmul(self._test_basis.transpose(), yn[self._sample_indices])

            # B. solve ROM
            rom = ROM(basis=self._basis, problem=problem, hyperreduction=self._hyperreduction, sample_indices=self._sample_indices, test_basis=self._test_basis, approx_mat=self._approx_mat)

            # D. Define residual function
            F = rom.createRightHandSide()
            def residual(x):
                _, v = rom.rightHandSide(np.matmul(self._test_basis, x),0.,np.matmul(self._test_basis, F))
                return v

            # Solve FOM
            qn = scipy.optimize.newton_krylov(residual, np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]), verbose=False, f_tol=1e-8)

            # Reconstruct solution
            yn = np.matmul(self._test_basis, qn)

            # Compute inverse of diagonal of Jacobian
            J = problem.createApplyJacobianResult(np.eye(np.shape(yn)[0]))
            problem.applyJacobian(yn, np.eye(np.shape(yn)[0]), 0., J)
            invJ = np.zeros(np.shape(J))
            count = 0
            for x in np.diag(J):
                invJ[count,count] = 1./x
                count += 1

            # Save results
            print(run_directory)
            np.savez('results.npz', y=yn, parameters=[parameter_sample['D'], parameter_sample['K']], res=np.matmul(self._test_basis,residual(qn)), invJ=invJ)

            # Swap back to base directory
            os.chdir(cdir)

            return 0 # NOTE: This function must return 0

        def compute_qoi(self, run_directory, parameter_sample):
            # Load from npz file and return y
            return np.load(run_directory + '/results.npz')['y']

        def compute_error_estimate(self, run_directory, parameter_sample):
            # Run model
            y = self.run_model(run_directory, parameter_sample)

            # Read in results
            dat = np.load(run_directory + '/results.npz')
            invJ = dat['invJ']
            res = dat['res']

            # Calculate error estimate
            return np.linalg.norm(np.matmul(invJ,res))

The ROM Model class is not yet initialized as it depends on the FOM training set to be completed first.

A separate class for the ROM was created for interfacing to Pressio Demo Apps. This class definition below would not ordinarily be required if the user is not using Pressio Demo Apps. ::

    class ROM():
        def __init__(self, basis, problem, hyperreduction, sample_indices, test_basis, approx_mat):
            self._basis = basis
            self._problem = problem
            self._hyperreduction = hyperreduction
            self._sample_indices = sample_indices
            self._test_basis = test_basis
            self._approx_mat = approx_mat
        
        def initializeRightHandSide(self):
            self._problem.createRightHandSide()
        
        def createRightHandSide(self):
            self._problem.createRightHandSide()
            return np.zeros(self._test_basis.shape[1])

        def rightHandSide(self, state, time, v):
            if self._hyperreduction == True:
                # NOTE: v is overwritten on call to rightHandSide
                v = np.matmul(self._approx_mat, v)
                self._problem.rightHandSide(state, time, v)
            else:
                self._problem.rightHandSide(state, time, v)
            # if self._hyperreduction == True:
            #     state = np.matmul(self._test_basis.transpose(), state[self._sample_indices])
            #     v = np.matmul(self._test_basis.transpose(), v[self._sample_indices])
            # else:
            #     state = np.matmul(self._basis.transpose(), state)
            #     v = np.matmul(self._basis.transpose(), v)
            state = np.matmul(self._test_basis.transpose(), state)
            v = np.matmul(self._test_basis.transpose(), v)

            return state, v

Defining the Parameter Space
***************
The final step is defining the parameter space, which is done by using the ParameterSpace class.

The sampling for the parameters is done using the Monte Carlo Sampler available through ROM Tools. ::

    class ParameterSpace():
        def __init__(self, parameter_name, num_parameters, bounds):
            self._parameter_name = parameter_name
            self._dimension = num_parameters
            self._bounds = np.array(bounds)

        def get_names(self):
            return self._parameter_name
        
        def get_dimensionality(self):
            return self._dimension
        
        def get_sampler(self):
            return rt.workflows.sampling_methods.MonteCarloSampler

        def generate_samples(self, n_samples):
            # Grab sampler
            sampler = self.get_sampler()

            # Generate samples
            samples = sampler(number_of_samples=n_samples, dimensionality=self._dimension, seed=1)

            # Scale to bounds
            scale =  self._bounds[:,1::] - self._bounds[:,0:1]
            samples = samples*scale.transpose() + self._bounds[:,0:1].transpose()
            
            return np.array(samples)

    # Define parameter space
    param_space = ParameterSpace(parameter_name=['K', 'D'], num_parameters=2, bounds=[[0.005, 0.015], [0.005, 0.015]])

Example: Monte Carlo Sampling of FOM
***************
Below is an example of sampling the FOM using Monte Carlo sampling. The sampling is done twice for a training set and for a test set and are stored in directories labeled "train" and "test", respectively. The figure shows a single snapshot from the training set of the FOM. ::

    # A. Run FOM at train/test points using montecarlo sampling
    rt.workflows.sampling.run_sampling(model=fom_model, parameter_space=param_space, run_directory_prefix='random/fom_', number_of_samples=n_snapshots, random_seed=1)
    rt.workflows.sampling.run_sampling(model=fom_model, parameter_space=param_space, run_directory_prefix='test/fom_', number_of_samples=n_test, random_seed=1)

    # B. Read in train/test snapshots (NOTE: snapshots should be a tensor)
    n_vars = 1 # number of PDE variables
    n = n_x * n_y # number of spatial DOFs
    snapshots_train = np.zeros((n_vars, n, n_snapshots))
    parameters_train = np.zeros((n_snapshots, 2))
    for i in range(0,n_snapshots):
        results = np.load('random/fom_' + str(i) + '/results.npz')
        snapshots_train[:,:,i] = results['y']
        parameters_train[i,:] = results['parameters']
    snapshots_test = np.zeros((n_vars, n, n_test))
    parameters_test = np.zeros((n_test, 2))
    for i in range(0,n_test):
        results = np.load('test/fom_' + str(i) + '/results.npz')
        snapshots_test[:,:,i] = results['y']
        parameters_test[i,:] = results['parameters']

    # C. Plot results for one snapshot
    plot_single_result(figure_path, mesh_obj, snapshots_train[0,:,0], x_label=f'$K={parameters_train[0,0]:.3f}$', y_label=f'$D={parameters_train[0,1]:.3f}$', suffix='_random')

The result of plot_single_result should look like below.

Creating the POD Basis
***************
This section shows an example of creating the POD basis. ::

    truncater = rt.vector_space.utils.truncater.NoOpTruncater()
    orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()
    pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots_train, truncater=truncater, orthogonalizer=orthogonalizer)
    basis = pod_space.get_basis()[0]
    np.savez('basis.npz',basis=basis) # NOTE: This is read in by run_model

Example: ROM trained on FOM snapshots using MC Sampling
***************
This section shows an example of running the ROM. ::
    
    # A. Calculate trial space from snapshots
    truncater = rt.vector_space.utils.truncater.NoOpTruncater()
    orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()
    pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots_train, truncater=truncater, orthogonalizer=orthogonalizer)
    basis = pod_space.get_basis()[0]
    np.savez('basis.npz',basis=basis) # NOTE: This is read in by run_model

    # B. Define ROM Model
    rom_model = ROM_Model(hyperreduction=False, offline_data_dir=os.getcwd())

    # B. Plot zeroth mode
    plot_single_result(figure_path, mesh_obj, basis[:,0], x_label=f'$K={parameters_train[0,0]:.3f}$', y_label=f'$D={parameters_train[0,1]:.3f}$', suffix='_basis0_random')

    # C. Run ROM at test points
    # NOTE: As long as the same seed is chosen, the ROM will run at the same points as the FOM.
    rt.workflows.sampling.run_sampling(model=rom_model, parameter_space=param_space, run_directory_prefix='test/rom_', number_of_samples=n_test, random_seed=1)

    # D. Read in ROM results
    rom_snapshots_test = np.zeros((n_vars, n, n_test))
    rom_parameters_test = np.zeros((n_test, 2))
    for i in range(0,n_test):
        results = np.load('test/rom_' + str(i) + '/results.npz')
        rom_snapshots_test[:,:,i] = results['y']
        rom_parameters_test[i,:] = results['parameters']

    # E. Plot ROM result at test point
    plot_single_result(figure_path, mesh_obj, rom_snapshots_test[0,:,0], x_label=f'$K={rom_parameters_test[0,0]:.3f}$', y_label=f'$D={rom_parameters_test[0,1]:.3f}$', suffix='_rom_random')

    # F. Plot ROM/FOM results at test point
    plot_results(figure_path, mesh_obj, snapshots_test[0,:,0], rom_snapshots_test[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_random')

    # G. Calculate error
    print('L2 Norm of Error between ROM and FOM at test points using MC sampling: ', np.linalg.norm(snapshots_test - rom_snapshots_test), '\n')