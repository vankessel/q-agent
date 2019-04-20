#!/usr/bin/env python3
import random
import gym
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from pathlib import Path
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act_value = env.action_space.sample()
            return act_value-5 if 6 <= act_value <= 9 else act_value
        act_values = self.model.predict(state)
        return max(enumerate(act_values[0]), key=lambda x: x[1])[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma
                          * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    AGENT_FILENAME = "mario.h5"
    EPISODES = 1000

    env = gym.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    state_size = env.observation_space.shape[0] * \
        env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = len(COMPLEX_MOVEMENT)

    agent = DQNAgent(state_size, action_size)

    # Set up save location
    base_dir = Path(__file__).parent.resolve()
    save_dir = base_dir / "../saves"
    try:
        save_dir = save_dir.resolve()
    except FileNotFoundError:
        save_dir.mkdir(parents=True)
        save_dir = save_dir.resolve()
        print("Made save path at: {}".format(save_dir))
    save_path = save_dir / AGENT_FILENAME

    if Path.is_file(save_path):
        print("Loading saved agent...")
        agent.load(save_path)

    done = False
    batch_size = 32

    for e in range(1, EPISODES + 1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time >= 500:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            time += 1
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(save_path)
