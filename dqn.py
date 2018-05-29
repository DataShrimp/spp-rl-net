import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

# Cart-Pole game environment
env = gym.make('CartPole-v0')

# test
env.reset()
print(env.action_space)
rl_info = []
for _ in range(20):
    # env.render()
    state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    rl_info.append([state, reward, done, info])
    if done:
        env.reset()


# RL Network
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()

        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(self.optimizer, 'mse')


# Memory to solve the rl stability problems
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]


# Hyper-parameter
train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

hidden_size = 16               # number of units in each Q-network hidden layer
learning_rate = 0.001         # Q-network learning rate

memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

# initial simulator
env.reset()
state, reward, done, _ = env.step(env.action_space.sample())
state = np.reshape(state, [1, 4])

# initial network
mainQN = QNetwork(learning_rate, hidden_size=hidden_size)

# initial memory
memory = Memory(memory_size)
for _ in range(pretrain_length):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])
    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state))

        env.reset()
        # Take one random step to forward
        state, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(state, [1, 4])
    else:
        memory.add((state, action, reward, next_state))
        state = next_state

# train
step = 0
for ep in range(1, train_episodes):
    total_reward = 0
    t = 0
    while t < max_steps:
        step += 1
        # env.render()

        # Explore or Exploit
        explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
        if explore_p > np.random.rand():
            # Make a random action
            action = env.action_space.sample()
        else:
            # Get action from Q-network
            Qs = mainQN.model.predict(state)[0]
            action = np.argmax(Qs)

        # Take action, get new state and reward
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward

        if done:
            # the episode ends so no next state
            next_state = np.zeros(state.shape)
            t = max_steps

            print('Episode: {}'.format(ep),
                  'Total reward: {}'.format(total_reward),
                  'Explore P: {:.4f}'.format(explore_p))

            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
            state = np.reshape(state, [1, 4])
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state
            t += 1

        # Replay
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))

        minibatch = memory.sample(batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
            # episode ends when state=0
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target_Q = mainQN.model.predict(next_state_b)[0]
                target = reward_b + gamma * np.amax(mainQN.model.predict(next_state_b)[0])
            targets[i] = mainQN.model.predict(state_b)
            targets[i][action_b] = target
        mainQN.model.fit(inputs, targets, epochs=1, verbose=0)


mainQN.model.save("dqn.h5")

if __name__ == "__main__":
    pass