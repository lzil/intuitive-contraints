# intuitive-constraints project

# requirements
- Python 3
- pymunk

# starter pack

To create something that looks like balls moving around under some constraints:
`python simulate.py -b 4 -c 3`

Wait for it to end, and you'll see that it saves the data from that run under `stimuli/XYZ_real.pkl`.

To load that data back up to run it again, run:
`python visualizer.py stimuli/XYZ_real.pkl`

To run this with collision and dynamic noise respectively, you can run:
`python visualizer.py stimuli/XYZ_real.pkl -n 1 3`

To run the Metropolis-Hastings algorithm to attempt to find the rest_length, stiffness and damping parameters of a spring, use:
`python proposer.py stimuli/XYZ_real.pkl`


# to build your own scene
Use `builder.py`.
Options `-f` (along with a file name) opens an already-saved `_real.pkl` file, and `-g` turns on gravity.
For instance,
`python builder.py -f XYZ_real.pkl -g`

Instructions:
- press `m` to iterate between different constraint types
- press `n` to iterate between shapes
- press `s` to save the scene in `stimuli/` (name of saved file should show in the console)
- press `r` to test run the scene
- press `g` to toggle gravity

- To create a constraint between two objects, click an object (it will turn _cadet blue_), then click on another object.
To cancel this selection, simply press the original object again.
- To create a constraint between an object and the ground, click an object (it will turn _cadet blue_), then click the location where you'd like to insert the constraint.
- To create shapes of different sizes, simply click and drag the mouse on an open area within the space.
- To move an object, simply drag it.
- To remove an object, right click it (this will remove any constraints as well)

Notes:
- the concept of a "space" has been abstracted away into a "Scene" class that is defined in `scene_tools.py`
- `scene_tools.py` also includes various useful helper functions, such a a gaussian and distance function, as well as the cost function