from typing import Any, Tuple, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip, vfx

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NOOP = -1


class RussellGrid(gym.Env):
    """
    The `RussellGrid` class defines a grid world environment with a specified map, reward matrix, action space,
    and observation space. The agent can take actions to move around the grid and receives rewards based on its
    actions and the state of the environment. The goal of the agent is to maximize its cumulative reward over
    a series of time steps.

    The environment is based on the OpenAI Gym interface, which allows it to be used with a variety of
    reinforcement learning algorithms.

    The environment is defined by a grid of cells, where each cell can be in one of three states: normal,
    terminal, or impossible. The agent can move between normal cells, but cannot move into terminal or
    impossible cells. The agent receives a reward of +1 for reaching a terminal cell and -1 for reaching an
    impossible cell. The agent receives a small negative reward (-0.04) for each time step it spends in a
    normal cell.

    The agent can take one of four actions at each time step: move up, move right, move down, or move left.
    The agent's actions are subject to a random deviation, which can cause the agent to move in a different
    direction than intended with a certain probability.

    The observation space is a discrete space that represents the current state of the agent. The state is
    represented by coordinates in the grid that can be converted to a single integer that corresponds to the ID 
    of the cell the agent is currently in using the static method "cell_id".

    The action space is a discrete space that represents the possible actions the agent can take. The action
    space has four possible actions: move up, move right, move down, or move left.

    The environment is initialized with a specified map, reward matrix, action space, and observation space.
    The agent's initial position is set using the `reset` method.

    The `reset` method sets the agent's position to a random empty cell on the map and returns the cell ID of
    the agent's new position.

    The `step` method takes an action as input, updates the agent's position based on the action and a random
    deviation, computes the observation, reward, and termination condition, and returns them.
    """

    def __init__(self, mode='python'):
        """
        The above function initializes a grid world environment with a specified map, reward matrix, action
        space, and observation space.
        """
        super().__init__()

        self.world_row = 3
        self.world_col = 4

        # Define map matrix for each cell: 1 means terminal state, 0 means normal state, -1 means impossible state
        # self.map should be:
        # array([[ 0.,  0.,  0.,  1.],
        #        [ 0., -1.,  0.,  1.],
        #        [ 0.,  0.,  0.,  0.]])
        self.map = np.zeros((self.world_row,self.world_col))
        self.map[0, 3] = 1
        self.map[1, 3] = 1
        self.map[1, 1] = -1

        # Define the reward matrix
        # self.reward_matrix should be:
        # array([[-0.04, -0.04, -0.04,  1.  ],
        #        [-0.04, -0.04, -0.04, -1.  ],
        #        [-0.04, -0.04, -0.04, -0.04]])
        self.reward_matrix = np.full((self.world_row,self.world_col), -0.04)
        self.reward_matrix[0, 3] = 1
        self.reward_matrix[1, 3] = -1

        # Define your action_space and observation_space here
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.world_col*self.world_row)

        self._agent_position = None

        self.mode = mode

        if self.mode == 'movie':
            # Initialize the visualization in case of use in jupyter notebook and wanting to store a movie
            self.myimages = []
            plt.ioff()
            self.fig2, self.ax2 = plt.subplots()
        else:
            # Initialize the visualization for python plots
            plt.ion()     
            self.fig2, self.ax2 = plt.subplots()


        # Creating a transition probability dictionary 'P' for each
        # state-action pair in the environment.
        # P[s][a] = list of pairs (prob, next_state) for each possible next state

        P = {}
        it = np.nditer(self.map, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[it.multi_index] = {a: [] for a in range(self.action_space.n)}

            ns_up = s if y == 0 else s - self.world_col
            ns_right = s if x == (self.world_col - 1) else s + 1
            ns_down = s if y == (self.world_row - 1) else s + self.world_col
            ns_left = s if x == 0 else s - 1

            # Following the gridworld example, if the next state is a wall, the agent stays in the same state
            if self.map[self.cell_coord(ns_up)] == -1:  
                ns_up = s
            if self.map[self.cell_coord(ns_right)] == -1:
                ns_right = s
            if self.map[self.cell_coord(ns_down)] == -1:
                ns_down = s
            if self.map[self.cell_coord(ns_left)] == -1:
                ns_left = s

            # For ease of computation, we define the states in multi_index format
            s = it.multi_index
            ns_up = self.cell_coord(ns_up)
            ns_right = self.cell_coord(ns_right)
            ns_down = self.cell_coord(ns_down)
            ns_left = self.cell_coord(ns_left)

            # Notice that actions are stochastic, so several next states possible with different probabilities
            # For example, if the agent goes UP, it will go UP with 0.8 probability, LEFT with 0.1 and RIGHT with 0.1
            # P[s][UP] = [(0.8, ns_up), (0.1, ns_left), (0.1, ns_right) ]
            # Notice state is represented as an integer (id_state), but you can use the function coord_id to convert to coordinates
            P[s][UP] = [(0.8, ns_up), (0.1, ns_left), (0.1, ns_right) ]
            P[s][RIGHT] = [(0.8, ns_right), (0.1, ns_up), (0.1, ns_down)]
            P[s][DOWN] = [(0.8, ns_down), (0.1, ns_left), (0.1, ns_right)]
            P[s][LEFT] = [(0.8, ns_left), (0.1, ns_up), (0.1, ns_down)]

            it.iternext()

        self.P = P



    @staticmethod
    def cell_id(coord, width=4):
        """
        The function `cell_id` calculates the cell ID based on the given coordinates and width of the grid.
        
        :param coord: The `coord` parameter is a tuple that represents the coordinates of a cell in the grid.
        It contains two elements: the row number and the column number of the cell
        :param width: The width parameter is an optional parameter that specifies the number of cells in
        each row of a grid. By default, it is set to 4, defaults to 4 (optional)
        :return: the cell ID, which is calculated by multiplying the first element of the coord tuple by the
        width and adding the second element of the coord tuple.
        """
        return coord[0] * width + coord[1]

    @staticmethod
    def cell_coord(cell_id, width=4):
        """
        The function `cell_coord` calculates the coordinates of a cell in the grid based on the given cell ID and width.
        
        :param cell_id: The `cell_id` parameter is an integer that represents the ID of a cell in the grid.
        :param width: The width parameter is an optional parameter that specifies the number of cells in
        each row of a grid. By default, it is set to 4, defaults to 4 (optional)
        :return: a tuple containing the row number and the column number of the cell in the grid.
        """
        row = cell_id // width
        col = cell_id % width
        return (row, col)


    def removeframes(self):
        """
        The function `removeframes` removes the frames stored in the myimages list.
        """
        self.myimages = []


    @property
    def agent_position(self):
        return self._agent_position

    @agent_position.setter
    def agent_position(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("agent_position must be a tuple of length 2")
        row, col = value
        if row < 0 or row >= self.map.shape[0] or col < 0 or col >= self.map.shape[1]:
            raise ValueError("agent_position must be within the bounds of the grid")
        self._agent_position = value



    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        The `reset` function sets the agent's position to a random empty cell on the map and returns the
        cell ID of the agent's new position.
        
        :param seed: The `seed` parameter is an optional integer that is used to seed the random number
        generator. By providing a specific seed value, you can ensure that the random numbers generated will
        be the same every time you run the code. This can be useful for reproducibility purposes
        :type seed: Optional[int]
        :param options: The `options` parameter is a dictionary that allows you to pass additional options
        or settings to the `reset` method. It is an optional parameter, meaning you can choose to provide it
        or not. If you choose to provide it, you can pass a dictionary containing any additional options or
        settings that you
        :type options: Optional[dict]
        :return: the cell ID of the agent's position.
        """
        super().reset(seed=seed)
        x,y = np.random.randint(0, 3), np.random.randint(0, 4)
        while self.map[x,y] != 0:
            x,y = np.random.randint(0, 3), np.random.randint(0, 4)
        self.agent_position = (x, y)
        return self.cell_id(self.agent_position) #, {}


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        The `step` function takes an action as input, updates the agent's position based on the action and a
        random deviation, computes the observation, reward, and termination condition, and returns them.
        
        Random deviation is set as "none", "left" or "rigth" with respect the direction taken by the agent with 
        probabilities of 0.8, 0.1, 0.1 repectively.

        New position has to be a valid position. If the new position is not valid with the action taken and deviation, 
        the agent remains in the same position.

        :param action: The `action` parameter represents the action taken by the agent. It is an integer
        value that corresponds to a specific action. In this code, the actions are represented as follows:
        :type action: int   {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
        :return: the following values:
        - observation: an ndarray representing the current state or observation of the agent.
        - reward: a float value representing the reward obtained by the agent for the current action.
        - done: a boolean value indicating whether the episode is finished or not.
        - False: a boolean value indicating whether the episode is finished or not (always False in this
        case).
        - {}: an empty dictionary of additional information (not used in this code).
        """

        deviation = ['no','left','right']
        dev = random.choices(deviation, weights=[0.8,0.1,0.1])[0]

        if action == 0:  #UP
            if dev == 'no':
                new_position = (self.agent_position[0]-1, self.agent_position[1])  
            elif dev == 'left':
                new_position = (self.agent_position[0], self.agent_position[1]-1) 
            elif dev == 'right':
                new_position = (self.agent_position[0], self.agent_position[1]+1)
        

        elif action == 1:  #RIGHT
            if dev == 'no':
                new_position = (self.agent_position[0], self.agent_position[1]+1) 
            elif dev == 'left':
                new_position = (self.agent_position[0]-1, self.agent_position[1]) 
            elif dev == 'right':
                new_position = (self.agent_position[0]+1, self.agent_position[1])     
                   
        elif action == 2:  #DOWN
            if dev == 'no':
                new_position = (self.agent_position[0]+1, self.agent_position[1]) 
            elif dev == 'left':
                new_position = (self.agent_position[0], self.agent_position[1]-1) 
            elif dev == 'right':
                new_position = (self.agent_position[0], self.agent_position[1]+1)             
        
        elif(action == 3): #LEFT 
            if dev == 'no':
                new_position = (self.agent_position[0], self.agent_position[1]-1) #LEFT
            elif dev == 'left':
                new_position = (self.agent_position[0]-1, self.agent_position[1]) 
            elif dev == 'right':
                new_position = (self.agent_position[0]+1, self.agent_position[1])     
            
        #Check if the new position is a valid position
        if (new_position[0]>=0 and new_position[0]<self.world_row):
                if(new_position[1]>=0 and new_position[1]<self.world_col):
                    if(self.map[new_position[0], new_position[1]] != -1):
                        self.agent_position = new_position

        observation = self.cell_id(self.agent_position)

        # Compute reward and terminate 
        reward = self.reward_matrix[self.agent_position[0], self.agent_position[1]]
        done = self.map[self.agent_position[0], self.agent_position[1]] == 1

        return observation, reward, done, False, {}


    def render(self):
        """
        The `render` function is used to visualize a map with an agent position marked as 'A' and grid
        lines.
        
        :param mode: The `mode` parameter is used to specify the rendering mode. It can take two values: 
        'movie' or 'python' - defaults to python (optional)
        :return: None.
        """
        self.ax2.imshow(self.map)
        self.ax2.text(self.agent_position[1], self.agent_position[0], 'A', color='black', ha='center', va='center',size=20)
        self.ax2.set_xticks(np.arange(0.5, self.map.shape[1], 1), labels = np.arange(self.map.shape[1]))
        self.ax2.set_yticks(np.arange(0.5, self.map.shape[0], 1), labels = np.arange(self.map.shape[0]))
        self.ax2.grid()#color='black', linestyle='-', linewidth=1)

        if self.mode == 'movie':
            self.myimages.append(mplfig_to_npimage(self.fig2))
        else:
            plt.pause(0.2) # Uncomment this to enable slow motion mode
        self.ax2.clear()
        return None
    

    def close(self):
        """
        The function `close` closes the matplotlib figure of the environment.
        """
        plt.close(self.fig2)


if __name__ == "__main__":
    # Create the environment
    env = RussellGrid()

    # Reset
    obs = env.reset()
    env.render()

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            env.reset()
            env.render()
    env.close()
