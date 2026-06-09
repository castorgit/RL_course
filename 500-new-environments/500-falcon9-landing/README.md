# Falcon9 Landing Notebooks

These notebooks use the local `Falcon9-v0` Gymnasium environment from the
`falcon9/` folder in this directory.

To run any notebook here, keep the `falcon9/` folder in the same location as the
notebook you open. The notebooks import `falcon9` directly and register the
environment with Gymnasium from that local package. If the folder is moved,
renamed, or missing, `gym.make("Falcon9-v0")` will not resolve correctly.

Start with [000_Falcon9_Random_Walk.ipynb](000_Falcon9_Random_Walk.ipynb). It
shows the complete environment invocation, documents all configurable
parameters, explains the observation and action spaces, runs a random-walk
episode, and visualizes the result with rendered frames.
