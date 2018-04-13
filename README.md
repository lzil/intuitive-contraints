# intuitive-constraints project

# requirements
- Python 3
- pymunk
- pymc3

To produce something that looks like balls moving around under some constraint:
`python multiple_ball.py spring -b 4 -v`

Wait for it to end, and you'll see that it saves the data from that run under `stimuli/spring/XYZ_real.pkl`.

To load that data back up to run it again, run:
`python data_loader.py stimuli/spring/XYZ_real.pkl -v`

To run this with collision and dynamic noise respectively, you can run:
`python data_loader.py stimuli/spring/XYZ_real.pkl -v -n 1 3`

To run the Metropolis-Hastings algorithm to attempt to find the rest_length, stiffness and damping parameters of a spring, use:
`python proposer.py stimuli/spring/XYZ_real.pkl`



Notes:
- the concept of a "space" has been abstracted away into a "Scene" class that is defined in `scene_tools.py`
- `scene_tools.py` also includes various useful helper functions, such a a gaussian and distance function, as well as the cost function
- the MH algorithm isn't doing that well right now. will need to work on cost function
- also need to work on creating more general constraints