import collections
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from environment import Environment
from visualiser import QValueVisualiser

Transition = collections.namedtuple("Transition", "state action reward next_state")


class Agent:
    def __init__(self, environment):
        # The distance moved by the agent in each step
        self.step_size = 0.1

        # Set the agent's environment
        self.environment = environment

        # Initialise the agent's memory using a replay buffer of max length 5000
        self.memory = ReplayBuffer(5000)

        # Initialise a deep Q-network for training our agent
        self.dqn = DQN()

        # Reset the agent's initial state and total reward
        self.reset()

    # Reset the environment and set the agent to its initial state
    def reset(self):
        # Reset the environment and set the agent's state to the initial state as defined by the environment
        self.state = self.environment.reset()

        # Set the agent's total reward for this episode to zero
        self.total_reward = 0.0

    # Make the agent take one step in the environment
    def step(self, epsilon):
        # Choose the next action
        discrete_action = self.__choose_next_action(epsilon)

        # Convert the discrete action into a continuous action
        continuous_action = self.__discrete_action_to_continuous(discrete_action)

        # Take one step in the environment, using this continuous action, based on the agent's current state.
        # This returns the next state, and the new distance to the goal from this new state.
        next_state, distance_to_goal = self.environment.step(
            self.state, continuous_action
        )

        # Compute the reward for this action
        reward = self.__compute_reward(distance_to_goal)

        # Create a transition tuple for this step
        transition = Transition(self.state, discrete_action, reward, next_state)

        # Set the agent's next state
        self.state = next_state

        # Update the agent's reward for this episode
        self.total_reward += reward

        return transition

    # Choose the agent's next action
    def __choose_next_action(self, epsilon):
        # Adopt a greedy policy with probability of 1 - epsilon
        if np.random.random() > epsilon:
            # Predict the Q-values for each action given the agent's current state
            q_values = self.dqn.q_network.forward(torch.tensor(self.state))

            return q_values.argmax().item()

        # Otherwise, randomly pick an action between 0 and 3
        return np.random.choice(4)

    # Convert a discrete action (as used by a DQN) to a continuous action (as used by the environment)
    def __discrete_action_to_continuous(self, discrete_action):
        # Return an action to move in [x, y] direction (0 -> right, 1 -> up, 2 -> left, 3 -> down)
        if discrete_action == 0:
            direction = [self.step_size, 0]

        elif discrete_action == 1:
            direction = [0, self.step_size]

        elif discrete_action == 2:
            direction = [-self.step_size, 0]

        elif discrete_action == 3:
            direction = [0, -self.step_size]

        return np.array(direction, dtype=np.float32)

    # Compute rewards based on the agent's distance to the goal
    def __compute_reward(self, distance_to_goal):
        return float(self.step_size * (1 - distance_to_goal))


class DQN:
    def __init__(self):
        # Create a Q-network and a target network, which predict the Q-value for a particular state
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network = Network(input_dimension=2, output_dimension=4)

        # Perform an initial update of the target network to ensure weights are the same
        self.update_target_network()

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how
        # big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Train the Q-network using a minibatch of transition tuples
    def train(self, minibatch, gamma):
        # Set all the gradients stored in the optimiser to zero
        self.optimiser.zero_grad()

        # Create tensors from the minibatch for states, actions, rewards and next states
        states, actions, rewards, next_states = zip(*minibatch)
        state_tensor = torch.tensor(states)
        next_state_tensor = torch.tensor(next_states)
        action_tensor = torch.tensor(actions)
        reward_tensor = torch.tensor(rewards)

        # Do a forward pass of the Q-network to obtain a prediction of the Q-value for each state
        state_prediction = self.q_network.forward(state_tensor)

        # Obtain the predicted Q-value for each transition's corresponding action in the minibatch
        predicted_reward = state_prediction.gather(
            1, action_tensor.unsqueeze(1)
        ).squeeze(1)

        # Do a forward pass of the target network to obtain a prediction of the Q-value for each next state
        next_state_prediction = self.target_network.forward(next_state_tensor)

        # Compute the actual reward by taking into account each next state's max predicted Q-value, discounted
        actual_reward = reward_tensor + gamma * next_state_prediction.detach().max(1)[0]

        # Calculate the loss for the minibatch of transitions
        loss = torch.nn.MSELoss()(predicted_reward, actual_reward)

        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters
        loss.backward()

        # Take one gradient step to update the Q-network
        self.optimiser.step()

        # Return the loss as a scalar
        return loss.item()

    # Update the target network by copying the weights of the Q-network
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Evaluate the Q-values for given samples of a discretised state space
    def evaluate_grid(self, grid_samples):
        rows, cols = grid_samples.shape[0], grid_samples.shape[1]

        # Initialise Q-values for each discrete action in each state
        q_values = torch.zeros((rows, cols, 4))

        # Loop over each cell in the discretised state grid
        for row in range(rows):
            for col in range(cols):
                # Predict the Q-values and average over the samples for each discrete action
                q_values[row, col] = self.q_network.forward(
                    grid_samples[row, col]
                ).mean(0)

        return q_values


class Network(torch.nn.Module):
    # Network takes as arguments the dimension of the network's input (i.e. the dimension of the state), and
    # the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()

        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(
            in_features=100, out_features=output_dimension
        )

    # Send some input data through the network and returns the network's output. In this example, a ReLU activation
    # function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)

        return output


class ReplayBuffer(collections.deque):
    def __init__(self, maxlen):
        super(ReplayBuffer, self).__init__(maxlen=maxlen)

    def sample(self, size):
        return random.sample(self, size)


# StateSpaceGrid discretises a state space into a grid of given length with dim x dim cells
class StateSpaceGrid:
    def __init__(self, length, dim):
        self.cell_length = length / dim
        self.dim = dim

    # Generate a sample of [x, y] coordinates from each cell in the discretised state grid
    def sample(self, sample_size):
        grid_samples = torch.zeros((self.dim, self.dim, sample_size, 2))

        for row in range(self.dim):
            for col in range(self.dim):
                x = col * self.cell_length
                y = (self.dim - row - 1) * self.cell_length

                grid_samples[row, col] = torch.tensor(
                    list(
                        zip(
                            np.random.uniform(x, x + self.cell_length, sample_size),
                            np.random.uniform(y, y + self.cell_length, sample_size),
                        )
                    )
                )

        return grid_samples


if __name__ == "__main__":
    environment = Environment(display=True, magnification=500)
    agent = Agent(environment)

    gamma = 0.9
    epsilon = 0.0
    epsilon_min = 0.0
    epsilon_decay = 0.995
    minibatch_size = 100
    update_frequency = 10

    episodes = 100
    steps = 100
    losses = [0] * episodes

    grid = StateSpaceGrid(length=1, dim=10)
    visualiser = QValueVisualiser(environment=environment, magnification=500)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(xlabel="Episodes", ylabel="Loss", title="Loss Curve for Deep Q-Network")

    for i in trange(episodes):
        # Reset the agent before the start of each episode
        agent.reset()

        for step in range(steps):
            # Take a step using an epsilon-greedy policy and obtain the transition tuple
            transition = agent.step(epsilon)

            # Add the transition to the agent's replay buffer
            agent.memory.append(transition)

            # Start training the Q-network once the replay buffer is full
            if len(agent.memory) > minibatch_size:
                # Sample a minibatch from the agent's replay buffer
                minibatch = agent.memory.sample(minibatch_size)

                # Train the deep Q-Network using the minibatch
                loss = agent.dqn.train(minibatch, gamma)

                # Every once in awhile
                if step % update_frequency == 0:
                    # Update the target network using the Q-network's weights
                    agent.dqn.update_target_network()

                    # Decay the value of epsilon so the agent does less exploration
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)

                # Add to the mean loss for this episode
                losses[i] += loss / steps

    # Collect samples from each cell in the discretised state grid
    grid_samples = grid.sample(100)

    # Evaluate the Q-values for each state-action in the grid using the samples
    q_values = agent.dqn.evaluate_grid(grid_samples)

    # Use a greedy policy to choose the actions with the highest Q-value
    greedy_policy = q_values.argmax(2)

    # Visualise the Q-values and greedy policy
    visualiser.draw_q_values(q_values.detach().numpy())
    environment.draw_policy(greedy_policy.detach().numpy(), steps)

    # Prune any losses prior to training the Q-network
    losses = list(filter(lambda loss: loss, losses))

    ax.plot(losses)
    plt.yscale("log")
    plt.show()
