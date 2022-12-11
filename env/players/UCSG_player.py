from baselines.common.policies import build_policy
from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
from gfootball.env import player_base
from gfootball.examples import models  
import gym
import joblib
import numpy as np
#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
print(tf.__version__)


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

    self._action_set = (env_config['action_set']
                        if 'action_set' in env_config else 'default')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    self._player_prefix = 'player_{}'.format(player_config['index'])
    stacking = 4 if player_config.get('stacked', True) else 1
    #policy = player_config.get('policy', 'cnn') # 取得player_config中key為policy的value, 如果找不到則輸入cnn
    self._stacker = ObservationStacker(stacking)

    policy = "gfootball_impala_cnn" 
    with tf.variable_scope(self._player_prefix):
      with tf.variable_scope('cnn_model'):
        # get register policy, build_policy(env,policy)
        policy_fn = build_policy(DummyEnv(self._action_set, False), policy)
        self._policy = policy_fn(nbatch=1, sess=self._sess)
    _load_variables(player_config['checkpoint'], self._sess,
                    prefix=self._player_prefix + '/')

  def __del__(self):
    self._sess.close()

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    # take action的observation為frame
    observation = observation_preprocessing.generate_smm(observation) # 在球員上標註點
    observation = self._stacker.get(observation)
    action = self._policy.step(observation)[0][0] # baseline function
    actions = [football_action_set.action_set_dict[self._action_set][action]]
    return actions

  def reset(self):
    self._stacker.reset()


def _load_variables(load_path, sess, prefix='', remove_prefix=True):
  """Loads variables from checkpoint of policy trained by baselines."""

  # Forked from address below since we needed loading from different var names:
  # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
  variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
               if v.name.startswith(prefix)]

  loaded_params = joblib.load(load_path)
  restores = []
  for v in variables:
    v_name = v.name[len(prefix):] if remove_prefix else v.name
    restores.append(v.assign(loaded_params[v_name]))

  sess.run(restores)


class ObservationStacker(object):
  """Utility class that produces stacked observations."""

  def __init__(self, stacking):
    self._stacking = stacking
    self._data = []

  def get(self, observation):
    if self._data:
      self._data.append(observation)
      self._data = self._data[-self._stacking:]
    else:
      self._data = [observation] * self._stacking
    return np.concatenate(self._data, axis=-1)

  def reset(self):
    self._data = []


class DummyEnv(object):
  # We need env object to pass to build_policy, however real environment
  # is not there yet.

  def __init__(self, action_set, stacking):
    self.action_space = gym.spaces.Discrete(
        len(football_action_set.action_set_dict[action_set]))
    self.observation_space = gym.spaces.Box(
        0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)
    # gym.spaces.box(low, high, shape, dtype)