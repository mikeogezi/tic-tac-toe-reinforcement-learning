import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

E = '_'
X = 'X'
O = 'O'

class InvalidMoveException(Exception):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

class WrongTurnException(Exception):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

class TicTacToeEnv (gym.Env):
  def __init__(self):
    super(TicTacToeEnv, self).__init__()

    self.set_game_type()
    self.learner, self.opponent = X, O
    self.grid = self.initial_grid
    self.player_to_int = {
      '_': 0,
      'X': 1,
      'O': -1
    }
    self.players = [ self.learner, self.opponent ]
    self.x_win = X * len(self.grid)
    self.o_win = O * len(self.grid)
    self.observation_space = spaces.Discrete(len(self.grid) ** 2)
    self.action_space = spaces.Discrete(len(self.grid) ** 2)
    self.reward_range = (-100, 50)
    self.info = {'x_wins': 0, 'o_wins': 0, 'draws': 0, 'invalidated_games': 0}
    self.steps = 0

  def set_game_type(self, game_type='learner_vs_computer'):
    self.is_lvp = game_type == 'learner_vs_player'

  def _next_observation(self):
    return list(map(lambda x: self.player_to_int[x], list(np.reshape(self.grid, (self.observation_space).n))))

  @property
  def initial_grid(self):
    return [ 
      [ E, E, E ],
      [ E, E, E ],
      [ E, E, E ]
    ]

  def _log_move_player(self):
    if self.current_player == self.opponent:
      print('Opponent ({}\'s) move'.format(self.current_player))
    elif self.current_player == self.learner:
      print('Learner ({}\'s) move'.format(self.current_player))
    print(20 * '-')

  def _make_move(self, player, choice):
    self._log_move_player()
    a = choice // len(self.grid)
    b = choice - (a * len(self.grid))

    if self.grid[a][b] != E:
      raise InvalidMoveException('Cannot play into occupied cell')
    
    self.grid[a][b] = player
    self.switch_player()

  def switch_player(self):
    self.current_player = X if self.current_player == O else O

  @property
  def _action_space(self):
    _action_space = []

    i = 0
    for row in self.grid:
      for cell in row:
        if cell == E:
          _action_space.append(i)
        i += 1
    
    return _action_space
    
  def done_winner_or_nth (self, done, winner):
    if done and winner == None:
      return 'No winner 😔\n'
    elif done and winner == self.learner:
      return 'Learner (\'{}\') wins 💪\n'.format(winner)
    elif done and winner == self.opponent:
      return 'Opponent (\'{}\') wins 😔\n'.format(winner)

    return ''

  def step(self, choice):
    # print('\n\nchoice:', choice)
    print()

    invalid_move = False
    done, winner = False, None
    
    # learner's turn
    if self.learner == self.current_player:
      try:
        self._make_move(self.current_player, choice)
        done, winner = self._is_game_finished()
        out = self.done_winner_or_nth(done, winner)
        self.render()
        print(out) if out else None
      except InvalidMoveException:
        invalid_move = True
        print('Learner made invalid move')

      # opponent's turn
      # skip both turns but heavily penalize learner and stop game if an invalid move was made
      if not done and not invalid_move:
        _play_choice = None
        if self.is_lvp:
          print('Positions')
          print(10 * '-')
          _len = len(self.grid)
          pos_list = range(_len ** 2)
          game_list = self._next_observation()
          occ_list = list(map(lambda x: '.' if abs(x[1]) != 0 else x[0], zip(pos_list, game_list)))
          print(pd.DataFrame(
            np.reshape(occ_list, (_len, _len))
          ))
          _play_choice = int(input('\nPlay your \'{}\', Your choices are {}: '.format(self.opponent, self._action_space)))
        else:
          _play_choice = np.random.choice(self._action_space)

        self._make_move(self.current_player, _play_choice)
        done, winner = self._is_game_finished()
        out = self.done_winner_or_nth(done, winner)
        self.render()
        print(out) if out else None
    else:
      raise WrongTurnException('Learner should be first')
    
    reward = self._compute_reward(done, winner, invalid_move)
    observation = self._next_observation()
    self.steps += 1

    if done:
      max_steps = math.ceil((len(self.grid) ** 2) / len(self.players))
      urgency_inv = max_steps + 1 - self.steps
      print('reward:', reward, '| max_steps:', max_steps, '| urgency_inv:', urgency_inv, '| steps:', self.steps)
      if winner == X:
        self.info['x_wins'] += 1
        reward *= urgency_inv
      elif winner == O:
        self.info['o_wins'] += 1
        reward *= urgency_inv
      elif winner == None:
        self.info['draws'] += 1
    
    if invalid_move:
      self.info['invalidated_games'] += 1

    return observation, reward, done if not invalid_move else True, self.info

  def _compute_reward(self, done, winner, invalid_move = False):
    reward = None

    # learner made invalid move
    if invalid_move:
      reward = -100
    elif done:
      # learner won
      if winner == self.learner:
        reward = 10
      # opponent won
      elif winner == self.opponent:
        reward = -10
      # no winner
      else:
        reward = -1
    else:
      # game still ongoing
      reward = 0

    return reward

  def reset(self):
    self.current_player = self.learner # np.random.choice(self.players)
    self.steps = 0
    self.grid = self.initial_grid
    return self._next_observation()

  def reset_info(self):
    self.info = {'x_wins': 0, 'o_wins': 0, 'draws': 0, 'invalidated_games': 0}

  def render(self, mode='human'):
    print(pd.DataFrame(self.grid))
    print()

  def close (self):
    print(self.info)

  def _is_game_finished (self):
    # horizontal wins
    for row in self.grid:
      __concat = ''.join(row)
      if __concat == self.o_win:
        return (True, O)
      elif __concat == self.x_win:
        return (True, X)

    # diagonal wins
    d_a, d_b = [], []
    for r in range(len(self.grid)):
      d_a.append(self.grid[r][r])
      r_ = -(r + 1)
      d_b.append(self.grid[r_][r])

    __concat_a, __concat_b = ''.join(d_a), ''.join(d_b)
    if __concat_a == self.o_win or __concat_b == self.o_win:
      return (True, O)
    elif __concat_a == self.x_win or __concat_b == self.x_win:
      return (True, X)

    # vertical wins
    for i in range(len(self.grid)):
      v = []
      for row in self.grid:
        v.append(row[i])
      __concat = ''.join(v)
      if __concat == self.o_win:
        return (True, O)
      elif __concat == self.x_win:
        return (True, X)

    # no winner, all cells are filled
    e = 0
    for row in self.grid:
      e += row.count(E)
    if e == 0:
      return (True, None)

    # no winner yet
    return (False, None)