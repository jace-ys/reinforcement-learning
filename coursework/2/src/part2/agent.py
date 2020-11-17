import collections
import random

import numpy as np
import torch


Transition = collections.namedtuple("Transition", "state action reward next_state")


class DQN:
    def __init__(self, lr):
        # Create a Q-network and a target network, which predict the Q-value for a particular state
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network = Network(input_dimension=2, output_dimension=4)

        # Perform an initial update of the target network using the Q-network's weights
        self.update_target_network()

        # Define the optimiser which is used when updating the Q-network. The learning rate determines
        # how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    # Train the Q-network on a minibatch of transition tuples using double deep Q-learning
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

        # Do a forward pass of the target network to obtain a prediction of the Q-value for each
        # next state and choose the corresponding action with the highest Q-value
        next_state_action_tensor = (
            self.target_network.forward(next_state_tensor).detach().argmax(1)
        )

        # Do a forward pass of the Q-network to obtain a prediction of the Q-value for each
        # next state and obtain the predicted Q-value for each transition's corresponding
        # action with the highest Q-value as predicted by the target network
        next_state_prediction = (
            self.q_network.forward(next_state_tensor)
            .gather(1, next_state_action_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Compute the actual reward by taking into account each next state's maximum predicted Q-value, discounted
        actual_reward = reward_tensor + gamma * next_state_prediction

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

    # Randomly sample, without replacement, a minibatch of the given size from the ReplayBuffer
    def sample(self, size):
        return random.sample(self, size)


class Agent:
    def __init__(self):
        # Initialise a deep Q-network for training the agent
        self.dqn = DQN(lr=0.001)

        # Initialise the agent's memory using a ReplayBuffer
        self.memory = ReplayBuffer(6000)

        # Store the latest state of the agent in the environment
        self.state = None

        # Store the latest action which the agent has applied to the environment
        self.action = None

        # Set the episode length
        self.episode_length = 300

        # Distance to move when taking an action
        self.step_size = 0.02

        # Initialise the total number of steps taken by the agent
        self.step_count = 0

        # Size of minibatch to sample from the ReplayBuffer for training the agent
        self.minibatch_size = 600

        # Value of discount factor to use when applying Bellman equation
        self.gamma = 0.9

        # Hyperparameters for decaying the value of epsilon over episodes
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.988

        # A flag for controlling how often to update the target network and value of epsilon
        self.update_frequency = 300

    # Check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        return self.step_count % self.episode_length == 0

    # Get the next action for the given state
    def get_next_action(self, state):
        # Adopt a greedy policy with probability of 1 - epsilon
        if np.random.random() > self.epsilon:
            # Select the action with the highest Q-value given the agent's current state
            q_values = self.dqn.q_network.forward(torch.tensor(state))
            action = q_values.argmax().item()
        else:
            # Randomly select a discrete action between 0 and 3
            action = np.random.choice(4)

        # Update the number of steps which the agent has taken
        self.step_count += 1

        # Store the state; this will be used later, when storing the transition
        self.state = state

        # Store the action; this will be used later, when storing the transition
        self.action = action

        # Convert the discrete action into a continuous one
        return self.__discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying self.action at self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # If the agent is nearing the end of an episode, assign a negative reward with small probability,
        # proportional to the distance from the goal. This helps the agent get out from being stuck in
        # local maxima due to sub-optimal policies.
        if (
            distance_to_goal > 0.03
            and self.step_count % self.episode_length > 280
            and np.random.random() < 0.05
        ):
            reward = float(-distance_to_goal)
        else:
            # Convert the distance to a reward
            reward = float(1 - distance_to_goal)

        # Create a transition and add it to the agent's memory
        transition = Transition(self.state, self.action, reward, next_state)
        self.memory.append(transition)

        # Sample a minibatch to train the deep Q-network on once the agent's memory is full
        if len(self.memory) > self.minibatch_size:
            minibatch = self.memory.sample(self.minibatch_size)
            self.dqn.train(minibatch, self.gamma)

        # Occasionally update the target network and decay the value of epsilon
        if self.step_count % self.update_frequency == 0:
            self.dqn.update_target_network()
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Select the action with the highest Q-value given the agent's current state
        q_values = self.dqn.q_network.forward(torch.tensor(state))
        action = q_values.argmax().item()

        # Convert the discrete action into a continuous one
        return self.__discrete_action_to_continuous(action)

    # Convert a discrete action (as used by a DQN) to a continuous action (as used by the environment)
    def __discrete_action_to_continuous(self, action):
        # Return an action to move in [x, y] direction (0 -> right, 1 -> up, 2 -> left, 3 -> down)
        if action == 0:
            direction = [self.step_size, 0]
        elif action == 1:
            direction = [0, self.step_size]
        elif action == 2:
            direction = [-self.step_size, 0]
        if action == 3:
            direction = [0, -self.step_size]

        return np.array(direction, dtype=np.float32)
