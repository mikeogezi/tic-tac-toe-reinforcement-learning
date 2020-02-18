import gym
from gym.envs.registration import register
import joblib
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

register(id='TicTacToe-v0', entry_point='envs:TicTacToeEnv')
env = gym.make('TicTacToe-v0')
env.set_game_type('learner_vs_player')
# env.set_game_type('learner_vs_computer')

model = joblib.load('model_pretrained.bin')

memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=-1.0, value_test=.05, nb_steps=1000)

dqn = DQNAgent(model=model, policy=policy, nb_actions=(env.action_space).n, memory=memory)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.test(env, nb_episodes=1, visualize=True)
