from keras.models import load_model
import gym
import numpy as np

model = load_model("dqn.h5")

env = gym.make('CartPole-v0')

rl_info = []
state = env.reset()
i = 0
while True:
    env.render()
    q = model.predict(state.reshape(1, 4))
    action = np.argmax(q)
    state, reward, done, info = env.step(action)
    rl_info.append([i, state, reward, done, info])
    i = i+1
    if done:
        print(rl_info)
        break

if __name__ == "__main__":
    pass
