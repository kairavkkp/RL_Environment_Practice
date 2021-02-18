from paddle_screen import Paddle

import random
import os
import numpy as np
from keras import Sequential
import json
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pickle
from json import JSONEncoder
from datetime import datetime, date


env = Paddle()
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def to_dict(self):
        result = {
            "action_space": self.action_space,
            "state_space": self.state_space,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            # "memory": list(self.memory)
        }
        return result

    def save_object(self, filename):

        if not os.path.exists(filename):
            os.makedirs(filename)

        # Serializing Numpy type object
        dict_to_save = json.dumps(
            self.to_dict(), cls=NumpyEncoder)

        self.model.save(filename+'/model.h5')
        # Overwrites any existing file.
        with open(filename+'/conf.json', 'w') as f:
            json.dump(dict_to_save, f)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * \
            (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []

    action_space = 3
    state_space = 5
    max_steps = 1000
    max_score = -100000

    agent = DQN(action_space, state_space)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        # if score > max_score:
        #     max_score = score
        #     agent.save_object(
        #         f'./models/paddle_model_ep_{e}_step_{i}_score_{score}')
    return loss


if __name__ == '__main__':

    ep = 10
    loss = train_dqn(ep)
    now = datetime.now()
    format_time = now.strftime("%H-%M-%S")
    today = date.today()
    format_date = today.strftime("%b-%d-%Y")
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    # plt.show()
    plt.savefig(f'loss_plot_{format_date}_{format_time}.png')
