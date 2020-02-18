import gym
from gym.envs.registration import register
import joblib

register(id='TicTacToe-v0', entry_point='envs:TicTacToeEnv')
env = gym.make('TicTacToe-v0')

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_model(state_size, num_actions):
  input = Input(shape=(1, state_size))
  x = Flatten()(input)
  x = Dense(128, activation='relu')(x)
  x = Dense(64, activation='relu')(x)
  x = Dense(32, activation='relu')(x)
  output = Dense(num_actions, activation='linear')(x)
  model = Model(inputs=input, outputs=output)
  model.summary()
  return model
    
memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=-1.0, value_test=.05, nb_steps=1000)

model = build_model((env.observation_space).n, (env.action_space).n)
dqn = DQNAgent(model=model, policy=policy, nb_actions=(env.action_space).n, memory=memory)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, 
  nb_steps=5000,
  visualize=False)
history = dqn.test(env, nb_episodes=100, visualize=True)
print(history)

joblib.dump(model, 'model.bin')
print('Saved model to disk...')