Step-by-step in Python
======================

This step-by-step tutorial goes over how to use the PRESSIO ROM tools and workflow scripts on the
`2D reaction diffusion PDE problem <https://pressio.github.io/pressio-demoapps/diffusion_reaction_2d.html>`_ from pressio-demoapps.
The full example file is located at `https://github.com/Pressio/rom-tools-and-workflows <https://github.com/Pressio/rom-tools-and-workflows>`_.

Mesh Generation
***************

The first step is setting up the problem. First, we generate the mesh.::

    def generate_mesh(pressio_file_path, mesh_path, figure_path, n_x, n_y):
        import os

        # generate mesh
        os.system('python3 ' + pressio_file_path + '/meshing_scripts/create_full_mesh_for.py --problem diffreac2d -n ' 
                    + str(n_x) + ' ' + str(n_y) + ' --outdir ' + mesh_path)

        # load mesh
        mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        return mesh_obj

    pressio_file_path = 'my_pressio_file_path'
    mesh_path = 'my_mesh_path'
    figure_path = 'my_figure_path'
    n_x = 50
    n_y = 50
    mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=n_x, n_y=n_y)

The mesh should look like this.


Next, we define our full-order model and our reduced-order model objects.::

    # Define FOM and ROM Model
    fom_model = FOM_Model()
    rom_model = ROM_Model(hyperreduction=False, offline_data_dir=os.getcwd())

We then define our parameter space. We define the names of our parameters, K and D, and set their bounds.::

    # Define parameter space
    param_space = ParameterSpace(parameter_name=['K', 'D'], num_parameters=2, bounds=[[0.005, 0.015], [0.005, 0.015]])

Finally, we define a truncater, orthogonalizer, and vector space for the ROM. We choose no truncation for the basis, a Euclidean $L^{2}$ orthogonalizer, and POD modes for the snapshots.::

    # A. Calculate trial space from snapshots
    truncater = rt.vector_space.utils.truncater.NoOpTruncater()
    orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()
    pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots_train, truncater=truncater, orthogonalizer=orthogonalizer)

To retrieve the basis, we use our vector space function. The basis is saved for further use in the code.::

    basis = pod_space.get_basis()[0]
    np.savez('basis.npz',basis=basis)

We are all finished with setup!

Example: Monte Carlo Sampling of the FOM.

We can easily run several parameter samples of the FOM using the built-in sampling function.::
    
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

Example: ROM trained on FOM snapshots using MC sampling.

Running the ROM is as simple as running the built-in sampling function.::

    # Run ROM at test points
    # NOTE: As long as the same seed is chosen, the ROM will run at the same points as the FOM.
    rt.workflows.sampling.run_sampling(model=rom_model, parameter_space=param_space, run_directory_prefix='test/rom_', number_of_samples=n_test, random_seed=1)

    # Read in ROM results
    rom_snapshots_test = np.zeros((n_vars, n, n_test))
    rom_parameters_test = np.zeros((n_test, 2))
    for i in range(0,n_test):
        results = np.load('test/rom_' + str(i) + '/results.npz')
        rom_snapshots_test[:,:,i] = results['y']
        rom_parameters_test[i,:] = results['parameters']

    # Calculate error
    print('L2 Norm of Error between ROM and FOM at test points using MC sampling: ', np.linalg.norm(snapshots_test - rom_snapshots_test), '\n')

**finish**
