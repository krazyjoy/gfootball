
import gfootball.env as football_env
import numpy as np
import gym
from gfootball.env import football_action_set


from gfootball.env import observation_preprocessing
import tensorflow as tf
import sonnet as snt

print("tensorflow version: ",tf.__version__)
tf.disable_v2_behavior()
# observation[0]['frame'].shape = (720,1280,3)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def network_fn(frame):
#   # Convert to floats.
    
#     print(frame.shape) # pixles: shape(72,96,3), frame: shape(720,960,3,1)
#     #tf.graph()
#     frame = tf.cast(frame, dtype= tf.float16)
#     frame /= 255
#     frame = tf.expand_dims(frame, axis = -1) # [720, 1280, 3, 1]
#     print(frame.shape.as_list())
 
#   #with tf.variable_scope('convnet'):

  
#     conv_out = frame
#     conv_layers = [(16,2), (16, 2), (32, 2), (32, 2)]
#     for i, (num_ch, num_blocks) in enumerate(conv_layers):
#       # Downscale.
#       print("num_channels", num_ch)
#       print("num_blocks",num_blocks)
#       conv_out = snt.Conv2D(num_ch, 4, stride=1, padding='SAME')(conv_out) # 4th dim = num_ch
#       print(conv_out.shape.as_list()) # [720, 1280, 3, 16], [720, 640, 2, 16], [720, 320, 1, 32]
#       conv_out = tf.nn.pool(
#           conv_out,
#           window_shape=[4, 4],
#           pooling_type='MAX',
#           padding='SAME',
#           strides=[3, 3]) # 2nd_dim =
#       print("for loop: ") 
#       print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720,320, 1, 16], [720, 160, 1, 32]
#       # Residual block(s).
#       for j in range(num_blocks):
#         #with tf.variable_scope('residual_%d_%d' % (i, j)):
#           block_input = conv_out
#           conv_out = tf.nn.relu(conv_out) 
#           print("relu1:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
#           print("Conv2D1:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = tf.nn.relu(conv_out)
#           print("relu2:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
#           print("Conv2D1:")
#           print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720, 160, 1, 32], [720, 80, 1, 32], 
#           conv_out += block_input

#     conv_out = tf.nn.relu(conv_out)
#     conv_out = snt.BatchFlatten()(conv_out)
#     print("batchFlatten:")
#     print(conv_out.shape.as_list()) # [720, 2560]
#     conv_out = snt.Linear(32)(conv_out) # define 2nd dim size
#     print("Linear:")
#     print(conv_out.shape.as_list())
#     conv_out = tf.nn.relu(conv_out) # [720, 256]

#     state = tf.reshape(conv_out, [-1])
    
#     #print("num of features:", n_features)
#     #print("shape of state", state.shape)
#     return state


class ObservationStacker(object):
    def __init__(self, stacking):
      self._stacking = stacking
      self._data = []

    def get(self, observation):
      sess = tf.compat.v1.Session()
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      observation = observation.eval(session = sess)

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
        len(action_set))
    print(self.action_space)
    # pixel size: 
    self.observation_space = gym.spaces.Box(
        0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)

    # observation["frame"]  size:
    # self.observation_space = gym.spaces.Box(
    #   0, 255, shape=[720, 1280, 3], dtype=np.uint8
    # )
    print(self.observation_space)

class policy_network(object):

    def __init__(self, build_policy):
        #Discrete(19)
        #Box(720, 1280, 3)
      
        self.policy = np.asarray([list(build_policy.observation_space.shape), list(build_policy.action_space.shape)])
        print("policy", self.policy)
    def step(self, stateid):
        actions = self.policy[stateid] # 這裡不知道怎麼寫

# 1. 本身維度就小的, 但具體如何downgrade部清楚的input: pixels: shape(72,96)
    
# 2. 仿造ppo寫法使用raw裡面的frame (1280,720,3)    

class Network():
    def __init__(
          self,
          frame,
          n_features,
          n_actions,
          sess,
          learning_rate=0.01,
          reward_decay=0.9,
          e_greedy=0.9,
          replace_target_iter=300,
          memory_size=500,
          batch_size=32,
          e_greedy_increment=None,
          output_graph=False,
  ):
        self.frame = frame
        self.n_actions = n_actions
        self.n_features = n_features
        self.sess = sess
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.frame = self.network_fn(frame)

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]



        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)


        self.cost_his = []

    def network_fn(self,frame):
        # Convert to floats.
        
        print(observation.shape) # pixles: shape(72,96,3), frame: shape(720,960,3,1)
        #tf.graph()
        frame = tf.cast(frame, dtype= tf.float16)
        frame /= 255
        frame = tf.expand_dims(frame, axis = -1) # [720, 1280, 3, 1]
        print(frame.shape.as_list())
    
        #with tf.variable_scope('convnet'):

    
        conv_out = frame
        conv_layers = [(16,2), (16, 2), (32, 2), (32, 2)]
        for i, (num_ch, num_blocks) in enumerate(conv_layers):
            # Downscale.
            #print("num_channels", num_ch)
            #print("num_blocks",num_blocks)
            conv_out = tf.nn.conv2d(num_ch, 4, stride=1, padding='SAME')(conv_out) # 4th dim = num_ch
            #print(conv_out.shape.as_list()) # [720, 1280, 3, 16], [720, 640, 2, 16], [720, 320, 1, 32]
            conv_out = tf.nn.pool(
                conv_out,
                window_shape=[4, 4],
                pooling_type='MAX',
                padding='SAME',
                strides=[3, 3]) # 2nd_dim =
            #print("for loop: ") 
            #print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720,320, 1, 16], [720, 160, 1, 32]
        # Residual block(s).
        for j in range(num_blocks):
            #with tf.variable_scope('residual_%d_%d' % (i, j)):
            block_input = conv_out
            conv_out = tf.nn.relu(conv_out) 
            #print("relu1:")
            #print(conv_out.shape.as_list()) # 
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            #print("Conv2D1:")
            #print(conv_out.shape.as_list()) # 
            conv_out = tf.nn.relu(conv_out)
            #print("relu2:")
            #print(conv_out.shape.as_list()) # 
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            #print("Conv2D1:")
            #print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720, 160, 1, 32], [720, 80, 1, 32], 
            conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        #print("batchFlatten:")
        #print(conv_out.shape.as_list()) # [720, 2560]
        conv_out = snt.Linear(32)(conv_out) # define 2nd dim size
        #print("Linear:")
        #print(conv_out.shape.as_list())
        conv_out = tf.nn.relu(conv_out) # [720, 256]

        state = tf.reshape(conv_out, [-1])
        
        #print("num of features:", n_features)
        #print("shape of state", state.shape)
        return state
        # total learning step
            

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        #s = s.eval(session = self.sess)
        #s_ = s_.eval(session = self.sess)
        #print("type s", type(s))
        #print("type s'", type(s_))
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        #observation = observation.eval(session = self.sess)
        #observation = observation[np.newaxis, :]
        observation = self.frame.eval(session = self.sess)
        print("observation type: ", type(observation))
        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :] # select a pair of past memory

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    stacking = 1
    step = 0
    env = football_env.create_environment(env_name="5_vs_5", 
    representation='pixels', render = True)

    observation = env.reset()
    # convert frame to nn
    #nn_network = network_fn(state[0]['frame'])
    #state = network_fn(observation) # [72,96]
    #n_features = state.shape[0]
    action_set = football_action_set.get_action_set({'action_set': 'default'})
    n_actions = len(action_set)

    RL = Network(observation,
                n_features = 2304, 
                n_actions = n_actions,
                sess = sess,
                learning_rate=0.01,
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=200,
                memory_size=2000)

    acc_reward = 0
    # store observation into a stack
    #
    while True:
      

        # RL choose action based on observation
        action = RL.choose_action(RL.frame)
        print("action", action)
        # RL take action and get next observation and reward
        observation_, reward, done, info = env.step(action)

        acc_reward += reward
        state = RL.frame
        next_state = RL.network_fn(observation_)
        print("next_state", type(next_state))
        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            print("acc_reward", acc_reward)
            break

        step += 1

        # end of game
        print('game over')
        env.destroy()
  # take an action, observe env and receive reward
        """
        stacker = ObservationStacker(stacking)
        while  True:
          next_state, rew, done, info = env.step(env.action_space.sample())
          #print(next_state[0]['frame'])
          # convert raw observation into frame
          # frame = observation_preprocessing.generate_smm(next_state)
          #new_nn_network = network_fn(next_state[0]['frame'])
          new_nn_network = network_fn(next_state)
          # store new frame
          frame_stack = stacker.get(new_nn_network)
          # print("frame_stack", frame_stack.shape) # frame_stack (1, 72, 96, 4) => pixel


          steps += 1
          if steps % 100 == 0:
            # print(obs, rew, done, info)
            break
          if done:
            break
          
          """





def state_dict():
    width = 720
    height = 256
    pixels = 256
    AllStates = np.zeros((width, height,pixels))
    return AllStates
    
def checkState(AllStates, state):
    

    for i in range(AllStates[0]):
        for j in range(AllStates[1]):
            k = state[i][j] # find each pixel value
            AllStates[i][j][k] += 1 # if it matches
# Initialization: t= 1


#建立dummy env 為pixel observation shape

# for phase k = 1,2,..., do

    # Initialize Phase

    # Update confidence set

    # Optimistic Planning

    # Execute Policies




