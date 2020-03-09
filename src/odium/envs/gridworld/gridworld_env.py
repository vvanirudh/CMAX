'''
Create a simple gridworld env
'''
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# 0: white; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0: [1.0, 1.0, 1.0], 1: [0.5, 0.5, 0.5],
          2: [0.0, 0.0, 1.0], 3: [0.0, 1.0, 0.0],
          4: [1.0, 0.0, 0.0], 6: [1.0, 0.0, 1.0],
          7: [1.0, 1.0, 0.0]}


class GridWorld(gym.GoalEnv):
    def __init__(self, grid, render=False, start_state=None, goal_state=None):
        super(GridWorld, self).__init__()
        # Create an empty grid
        # A 10x10 grid with all free cells (represented by zero)
        self.size = grid.shape[0]
        self.grid = grid.copy()
        self.render = render
        # Define spaces
        # Action space is simply an element of {0, 1, 2, 3}
        # 0 - move left, 1 - move up, 2 - move down, 3 - move right
        # left-right is X direction, up-down is Y direction
        # But if its in a slip state, then
        # 0 - move right, 1 - move up, 2 - move down, 3 - move left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.MultiDiscrete([self.size, self.size]),
            achieved_goal=spaces.MultiDiscrete([self.size, self.size]),
            observation=spaces.MultiDiscrete([self.size, self.size]),
        ))

        self.start_state = start_state
        if start_state is None:
            self.start_state = np.array([0, 0])
        self.goal_state = goal_state
        if goal_state is None:
            self.goal_state = np.array([self.size-1, self.size-1])
        self.current_state = self.start_state.copy()

        self.np_random = np.random.RandomState()

        self.transition_dict = {}

        if render:
            # Visualization
            self.fig = plt.figure(0)
            self.img = self._gridmap_to_image()
            self.discretization = self.img.shape[0] // self.size
            plt.show(block=False)
            plt.axis('off')
            self._render()

    def _gridmap_to_image(self):
        img = np.zeros((128, 128, 3), dtype=np.float32)
        discretization = 128 // self.size

        # Get the grid
        for i in range(self.size):
            for j in range(self.size):
                img[i*discretization:(i+1)*discretization, j *
                    discretization:(j+1)*discretization] = np.array(COLORS[self.grid[i, j]])

        return img

    def _render(self):
        if not self.render:
            return
        img = self.img.copy()
        discretization = self.discretization
        # Mark the goal state
        i, j = self.goal_state
        img[i*discretization:(i+1)*discretization, j *
            discretization:(j+1)*discretization] = np.array(COLORS[3])

        # Mark the current state
        i, j = self.current_state
        img[i*discretization:(i+1)*discretization, j *
            discretization:(j+1)*discretization] = np.array(COLORS[4])

        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.0000001)
        return

    def reset(self):
        self.current_state = self.start_state.copy()
        self._render()
        return self.get_observation(self.current_state)

    def get_displacement(self, state, action):
        if self.grid[tuple(state)] == 0:
            # Free state
            if action == 0:
                return np.array([-1, 0])
            elif action == 1:
                return np.array([0, 1])
            elif action == 2:
                return np.array([0, -1])
            elif action == 3:
                return np.array([1, 0])
            else:
                raise Exception('Invalid action')

        elif self.grid[tuple(state)] == 2:
            # Slip state
            if action == 0:
                return np.array([1, 0])
            elif action == 1:
                return np.array([0, 1])
            elif action == 2:
                return np.array([0, -1])
            elif action == 3:
                return np.array([-1, 0])
            else:
                raise Exception('Invalid action')

        else:
            # Obstacle
            raise Exception('the agent is inside obstacle!')

    def get_actions(self):
        return np.arange(4)

    def check_goal(self, current_state, goal_state):
        if np.array_equal(current_state, goal_state):
            return True
        return False

    def get_observation(self, state):
        observation = {}
        observation['observation'] = state.copy()
        observation['achieved_goal'] = state.copy()
        observation['desired_goal'] = self.goal_state.copy()
        return observation

    def step(self, action):
        # Any action would move the agent one cell in a four-connected grid
        if self.current_state.tobytes() in self.transition_dict and action.tobytes() in self.transition_dict[self.current_state.tobytes()]:
            next_state = self.transition_dict[self.current_state.tobytes(
            )][action.tobytes()]
        else:
            next_state = self.current_state + \
                self.get_displacement(self.current_state, action)

            if self.out_of_bounds(next_state) or self.in_collision(next_state):
                # No displacement
                next_state = self.current_state.copy()

        cost = self.cost(next_state, self.goal_state)

        observation = self.get_observation(next_state)
        self.current_state = next_state.copy()
        self._render()

        return observation, cost, False, {}

    def out_of_bounds(self, state):
        if np.any(state < 0) or np.any(state >= self.size):
            return True
        return False

    def in_collision(self, state):
        if self.grid[tuple(state)] == 1:
            # Obstacle
            return True
        return False

    def render(self):
        current_grid = self.grid.astype(int).copy()
        current_grid = current_grid.astype(str)
        current_grid[self.current_state[0], self.current_state[1]] = 'A'
        current_grid[self.goal_state[0], self.goal_state[1]] = 'G'
        return np.array2string(current_grid, max_line_width=self.size*5, threshold=self.size**2+10)

    def get_neighbors(self, state, ignore_action=None):
        neighbors = []
        for ac in self.get_actions():
            if ignore_action is not None and ignore_action == ac:
                continue
            neighbor = state + self.get_displacement(state, ac)
            if self.out_of_bounds(neighbor) or self.in_collision(neighbor):
                continue
            neighbors.append((ac, neighbor))

        return neighbors

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)
        return

    def cost(self, state, goal_state):
        if self.check_goal(state, goal_state):
            return 0
        return 1

    def update_true(self, state, action, next_state):
        predicted_next_state = state + self.get_displacement(state, action)
        if self.out_of_bounds(predicted_next_state) or self.in_collision(predicted_next_state):
            predicted_next_state = state.copy()
        if not np.array_equal(predicted_next_state, next_state):
            # Need to update the model
            # Handle it by a case-to-case basis
            # What if predicted_next_state was in fact an obstacle?
            if np.array_equal(state, next_state):
                # if the next state and the current state are equal, it can only be
                # that there is an obstacle or it is out of bounds, but out of bounds
                # has already been taken care of, so it has to be an obstacle
                self.grid[tuple(predicted_next_state)] = 1
            else:
                # next_state and predicted_next_state are different and next_state is
                # not the same as state. So, the current state has to be a slip state
                self.grid[tuple(state)] = 2

        return True

    def update(self, state, action, next_state):
        # Store this transition, so that the next time it happens we execute this
        if state.tobytes() not in self.transition_dict:
            self.transition_dict[state.tobytes()] = {}
            self.transition_dict[state.tobytes(
            )][action.tobytes()] = next_state
        else:
            self.transition_dict[state.tobytes(
            )][action.tobytes()] = next_state


class EmptyGridWorld(GridWorld):
    def __init__(self, size=10, render=False):
        grid = np.zeros((size, size))
        super(EmptyGridWorld, self).__init__(grid=grid, render=render)


class ObstacleGridWorld(GridWorld):
    def __init__(self, size=10, render=False):
        grid = np.zeros((size, size))
        # Define a wall at y = int(size/2) and stretching from x = int(size/4) to size
        grid[int(size/4):size, int(size/2)] = 1
        super(ObstacleGridWorld, self).__init__(grid=grid, render=render)


class RandomObstacleGridWorld(GridWorld):
    def __init__(self, size=10, render=False, incorrectness=0.5):
        grid = np.random.choice([0, 1], size=(size, size), p=(
            1 - incorrectness, incorrectness))
        # Sample start and goal states randomly
        start_state = np.random.randint(size, size=(2,))
        goal_state = np.random.randint(size, size=(2,))
        while (start_state[0] >= goal_state[0]) or (start_state[1] >= goal_state[1]) or (np.abs(start_state - goal_state).sum() < 10):
            start_state = np.random.randint(size, size=(2,))
            goal_state = np.random.randint(size, size=(2,))
        # Construct a single path from start to goal
        row = start_state[0]
        column = start_state[1]
        while (row != goal_state[0]) or (column != goal_state[1]):
            # Toss a coin to either move in rows or columns
            toss = np.random.random()
            if toss < 0.5 and row != goal_state[0]:
                # Rows
                # Sample until what row to clear the path to
                end_row = np.random.randint(low=row+1, high=goal_state[0]+1)
                grid[row:end_row, column] = 0
                row = end_row
            elif toss >= 0.5 and column != goal_state[1]:
                # Columns
                # Sample until what column to clear the path to
                end_column = np.random.randint(
                    low=column+1, high=goal_state[1]+1)
                grid[row, column:end_column] = 0
                column = end_column

        # Start and goal should be free
        grid[tuple(start_state)] = 0
        grid[tuple(goal_state)] = 0

        super(RandomObstacleGridWorld, self).__init__(
            grid=grid, render=render, start_state=start_state, goal_state=goal_state)


class RandomSlipGridWorld(GridWorld):
    def __init__(self, size=10, render=False, incorrectness=0.5):
        grid = np.random.choice([0, 2], size=(
            size, size), p=(1 - incorrectness, incorrectness))
        # Sample start and goal states randomly
        start_state = np.random.randint(size, size=(2,))
        goal_state = np.random.randint(size, size=(2,))
        while (start_state[0] >= goal_state[0]) or (start_state[1] >= goal_state[1]) or (np.abs(start_state - goal_state).sum() < 10):
            start_state = np.random.randint(size, size=(2,))
            goal_state = np.random.randint(size, size=(2,))
        # Construct a single path from start to goal
        row = start_state[0]
        column = start_state[1]
        while (row != goal_state[0]) or (column != goal_state[1]):
            # Toss a coin to either move in rows or columns
            toss = np.random.random()
            if toss < 0.5 and row != goal_state[0]:
                # Rows
                # Sample until what row to clear the path to
                end_row = np.random.randint(low=row+1, high=goal_state[0]+1)
                grid[row:end_row, column] = 0
                row = end_row
            elif toss >= 0.5 and column != goal_state[1]:
                # Columns
                # Sample until what column to clear the path to
                end_column = np.random.randint(
                    low=column+1, high=goal_state[1]+1)
                grid[row, column:end_column] = 0
                column = end_column

        # Start and goal should be free
        grid[tuple(start_state)] = 0
        grid[tuple(goal_state)] = 0

        super(RandomSlipGridWorld, self).__init__(
            grid=grid, render=render, start_state=start_state, goal_state=goal_state)


def make_gridworld_env(env='empty', size=10, render=False, incorrectness=0.5):
    if env == 'empty':
        return EmptyGridWorld(size, render)
    elif env == 'obstacle':
        return ObstacleGridWorld(size, render)
    elif env == 'random_obstacle':
        return RandomObstacleGridWorld(size, render, incorrectness)
    elif env == 'random_slip':
        return RandomSlipGridWorld(size, render, incorrectness)
    else:
        raise NotImplementedError
