# 1. Problem definition
"""
1. Problem definition
In the reinforcement learning framework for this case study, the algorithm takes an action (buy, sell, or hold)
depending on the current state of the stock price. The algorithm is trained using a deep Q-learning model to perform
the best action. The key components of the reinforcement learning framework are:

- Agent : Trading agent.
- Action : Buy, sell, or hold.
- Reward function : Realized profit and loss (PnL) is used as the reward function for this case study.
The reward depends on the action: sell (realized profit and loss), buy (no reward), or hold (no reward).
- State : A sigmoid function of the differences of past stock prices for a given time window is used as the state.
- Environment : Stock exchange or the stock market.

Please refer to the website below for further information about RL and Algorithmic trading:
quantitative-research-trading.com

The data that will be used in this case study is the one of NASDAQ 100 closing prices. The data is extracted from Yahoo
Finance and contains ten years of daily data from 2012 to 2022.
"""

# 2. Data loading and Python packages

# 2.1.1. Packages for reinforcement learning
import keras
from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 2.1.2. Packages/modules for data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
import datetime
import math
from numpy.random import choice
import random
from collections import deque

# Diable the warnings
import warnings
warnings.filterwarnings('ignore')

# Display
set_option('display.width', 100)
set_option('display.max_rows', 500)
set_option('display.max_columns', 500)

# 2.2. Loading the data.
"""
The fetched data for the time period of 2012 to 2022 is loaded: 
"""
dataset = read_csv('NasdaqData.csv', index_col=0)
print('dataset type :', type(dataset))

# 3. Exploratory Data Analysis

# 3.1. Shape and description of data
print('shape of the dataset :', dataset.shape)
set_option('precision', 3)
print('Nasdaq data \n', dataset.tail())
print('Description of dataset \n', dataset.describe())

# 3.2. Data visualization - close price
dataset['Close'].plot()
plt.title('Nasdaq composite - close price')
plt.legend()
plt.savefig('Nasdaq composite - close price.png')
plt.show()

# 4. Data Preparation
# 4.1. Data Cleaning
"""
Let us check for the NAs in the rows, either drop them or fill them with the mean of the column.
"""
# Checking for any null values and removing the null values
print('Null Values =', dataset.isnull().values.any())

"""
In case there are null values fill the missing values with the last value available in the dataset.
"""
# Fill the missing values with the last value available in the dataset.
dataset = dataset.fillna(method='ffill')
print('Nasdaq data \n',dataset.head(2))

# 5. Evaluate Algorithms and Models
# 5.1. Train Test Split
"""
For the train and test split, it's usually about using 80% of dataset to train the model and 20% for testing.
"""

X = list(dataset["Close"])
X = [float(x) for x in X]

validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

# 5.2. Implementation of the algorithm
"""
- Agent Class: An object of the agent class is created using the training phase and it's used for the training the model.
- Helper functions : 
- Training module
"""

# 5.3. Agent class
"""
It consists of the following components:
• Constructor : init function with all important variables
• Function model
• Function act
• Function expReplay
"""


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        # State size depends and is equal to the window size, n previous days
        self.state_size = state_size  # normalized previous days,
        self.action_size = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.epsilon_decay = 0.9

        # self.model = self._model()

        self.model = load_model(model_name) if is_eval else self._model() # The function model is a deep learning model
        # that maps the states to actions.

    # Deep Q Learning model- returns the q-value when given state as input
    def _model(self):
        model = Sequential()
        # Input Layer
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        # Hidden Layers
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        # Output Layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    """
     Return the action on the value function With probability (1-$\epsilon$) choose the action which has the highest 
     Q-value.
     With probability ($\epsilon$) choose any action at random.
     Initially high epsilon-more random, later less 
     The trained agents were evaluated by different initial random condition and an e-greedy policy with epsilon 0.05. 
     This procedure is adopted to minimize the possibility of over-fitting during evaluation.
    """

# The function act returns an action given a state.
# It uses the model function and returns a buy, sell, or hold action:
    def act(self, state):
        # If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        # actions suggested.
        if not self.is_eval and random.random () <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        # set_trace()
        # action is based on the action that has the highest value from the q-value function.
        return np.argmax(options[0])

# The function expReplay is the key function, where the neural network is trained based on the observed experience.
    """
    This function implements the Experience replay mechanism, which stores a history of state, action, reward, and next 
    state transitions that are experienced by the agent.
    """
    def expReplay(self, batch_size):
        mini_batch = []
        l = len ( self.memory )
        for i in range ( l - batch_size + 1, l ):
            mini_batch.append ( self.memory[i] )

        # the memory during the training phase.
        for state, action, reward, next_state, done in mini_batch:
            target = reward  # reward or Q at time t
            # update the Q table based on Q table equation
            # set_trace()
            if not done:
                # set_trace()
                # max of the array of the predicted.
                target = reward + self.gamma * np.amax ( self.model.predict ( next_state )[0] )

                # Q-value of the state currently from the table
            target_f = self.model.predict ( state )
            # Update the output Q table for the given action in the table
            target_f[0][action] = target
            # train and fit the model where state is X and target_f is Y, where the target is updated.
            self.model.fit ( state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 5.4 Helper functions
# prints formatted price


def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# # returns the vector containing stock data from a fixed file
# def getStockData(key):
#     vec = []
#     lines = open("data/" + key + ".csv", "r").read().splitlines()

#     for line in lines[1:]:
#         vec.append(float(line.split(",")[4])) #Only Close column

#     return vec


# returns the sigmoid
def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))


# Returns an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    # block is which is the for [1283.27002, 1283.27002]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


# Plots the behavior of the output
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize = (15,5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label = 'Buying signal', markevery = states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label = 'Selling signal', markevery = states_sell)
    plt.title('Total gains: %f'%(profit))
    plt.legend()
    plt.savefig('Total gains: %f'%(profit)+'.png')
    plt.show()


window_size = 1
data = X_train
agent = Agent(window_size)
l = len(data) - 1
batch_size = 10
states_sell = []
states_buy = []
episode_count = 3


for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    # 1-get state
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        # 2-apply best action
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append(data[t])
            states_buy.append(t)
            print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            # 3: Get Reward
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            states_sell.append(t)
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        # 4: get next state to be used in bellman's equation
        next_state = getState(data, t + 1, window_size + 1)
        # 5: add to the memory
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            # Chart to show how the model performs with the stock going up and down for each
            plot_behavior(data, states_buy, states_sell, total_profit)

        # 6: Run replay buffer function
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))

# 6. Testing the data
# agent is already defined in the training set above.
#agent is already defined in the training set above.
test_data = X_test
l_test = len(test_data) - 1
state = getState(test_data, 0, window_size + 1)
total_profit = 0
is_eval = True
done = False
states_sell_test = []
states_buy_test = []
# Get the trained model
#model_name = "model_ep"+str(episode_count)
model_name = "saved_model.pb"
agent = Agent(window_size, is_eval, model_name)
state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range ( l_test ):
    action = agent.act ( state )
    # print(action)
    # set_trace()
    next_state = getState(test_data, t + 1, window_size + 1 )
    reward = 0

    if action == 1:
        agent.inventory.append(test_data[t])
        states_buy_test.append(t)
        print("Buy: " + formatPrice (test_data[t]))

    elif action == 2 and len (agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(test_data[t] - bought_price, 0)
        # reward = test_data[t] - bought_price
        total_profit += test_data[t] - bought_price
        states_sell_test.append(t)
        print("Sell: " + formatPrice(test_data[t]) + " | profit: " + formatPrice(test_data[t] - bought_price))

    if t == l_test - 1:
        done = True
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("------------------------------------------")
        print("Total Profit: " + formatPrice ( total_profit))
        print("------------------------------------------")

plot_behavior(test_data, states_buy_test, states_sell_test, total_profit)