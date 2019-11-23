# dnn-community-detection
Community detection in deep nerual networks

This project has the following dependences:

 - standard anaconda3 environment, and
 - graph-tool=2.29 (installed from conda-forge)
 - tensorflow=2.0.0
 - matplotlib=2.2.4
 
Notes on getting graph-tool drawing to work:

 - matplotlib version **must** be < 3.0; use 2.2.4 as specified above
 - For some plotting features, GTK is required. Run the following commands to get it working:

    conda install -c conda-forge pygobject
    conda install -c pkgw/label/superseded gtk3

To make sure it's working: `from graph_tool.draw import draw_hierarchy` should succeed with no warnings.

