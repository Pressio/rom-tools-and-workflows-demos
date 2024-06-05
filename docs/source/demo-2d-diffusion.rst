2D Reaction-Diffusion Problem
=============================

This step-by-step tutorial goes over how to use the PRESSIO ROM tools and workflow scripts on the
`2D reaction diffusion PDE problem <https://pressio.github.io/pressio-demoapps/diffusion_reaction_2d.html>`_ from pressio-demoapps.
The full example file is located at `https://github.com/Pressio/rom-tools-and-workflows <https://github.com/Pressio/rom-tools-and-workflows>`_.

Modules
***************

These are the modules that you will need to load for this problem.

.. jupyter-execute::
    :hide-output:

    %matplotlib ipympl

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

.. jupyter-execute::
    
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
    pressio_file_path = 'my_pressio_file_path'
    mesh_path = 'my_mesh_path'
    figure_path = 'my_figure_path'

    pressio_file_path = '/Users/ekrath/codes/pressio/pressio-demoapps'
    mesh_path = '/Users/ekrath/codes/pressio/mywork/demo-apps/2d-reaction-diffusion/mesh'
    figure_path = 'figs'

    # Generate and plot mesh
    fig = plt.figure(figsize=(10,7))
    fig.set_facecolor('w')
    ax = plt.gca()
    mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=25, n_y=25)
    plt.show()

    # NOTE: This is specific to the jupyter notebook only.
    # Interactable widget for changes in x and y mesh resolution.
    def update(n_x, n_y):
        mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=n_x, n_y=n_y)
        fig.canvas.draw()
        return mesh_obj
    f = ipyw.interactive(update, n_x=(5,50,1), n_y=(5,50,1))

    # Display the interactive widget
    display(f)



