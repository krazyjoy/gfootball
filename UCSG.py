import itertools
import math
from tkinter import N
import numpy as np
import gfootball.env as football_env
from absl import flags
import tensorflow as tf

from baselines.common.policies import build_policy

env = football_env.create_environment(env_name='5_vs_5', representation='pixels', render='True', number_of_left_players_agent_controls=5 )

# convolutionize env 
# build policy - default shape policy before looking the environment
# execute step and watch next observation
# convert observation to nn network

from gfootball.env import football_action_set
from baselines.common.models import register
import sonnet as snt

# 註冊一個新的policy nework, 需要透過cmd下指令
# python -m gfootball.play_game --players "UCSG_player:left_players=1,checkpoint= ~\Desktop\checkpoint,policy=cnn2"


@register("cnn2")
def gfootball_impala_cnn():
  def network_fn(frame):
    # Convert to floats.
    frame = tf.to_float(frame)
    frame /= 255
    with tf.variable_scope('convnet'):
      conv_out = frame
      conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
      for i, (num_ch, num_blocks) in enumerate(conv_layers):
        # Downscale.
        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
        conv_out = tf.nn.pool(
            conv_out,
            window_shape=[3, 3],
            pooling_type='MAX',
            padding='SAME',
            strides=[2, 2])

        # Residual block(s).
        for j in range(num_blocks):
          with tf.variable_scope('residual_%d_%d' % (i, j)):
            block_input = conv_out
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
            conv_out += block_input

    conv_out = tf.nn.relu(conv_out)
    conv_out = snt.BatchFlatten()(conv_out)

    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    return conv_out

  return network_fn



# initial phase
# Update the confidence set
# Optimistic Planning
# Execute Policies

#inner maximization for EVI
def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank): 

    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa




def UCSG(n_states, n_actions, T, delta):
    t = 1
    # Initial state
    vk = np.zeros((n_states, n_actions)) #vk(s,a)
    total_numbers = np.zeros((n_states, n_actions, n_states)) # nk(s,a,s')
    total_rewards = np.zeros(n_states, n_actions) #maximin
    nk = np.ones((n_states, n_actions)) #nk(s,a)
    #initial state
    st = env.reset()
    # Initialize phase k
    for k in itertools.count():
        t_k = t
        #Per-phase visitations
        #在之後的execute policy做係數update
        #compute estimates
        # delta:我們自己設定的容許誤差值    
        delta_1 = delta/(2*n_states*n_states*n_actions*np.log2(T))
        p_hat = vk.reshape((n_states, n_actions, 1))/n_states  #拆解

        # Update the confidence set
        confidence_bound_1 = np.clip(np.sqrt((2*n_states*np.log(1/delta)/n_states)), 0, 1) 
        p_tilde = confidence_bound_1 + p_hat
        #???
        confidence_bound_2_2 = min(np.sqrt(np.log(6/delta_1)/(2*n_states)), np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/n_states) + 7*np.log(6/delta_1)/(3*(n_states-1)))
        confidence_bound_2_1 = np.sqrt(2*np.log(6/delta_1)/(n_states-1))

        
        # alpha, gamma
        #Optimistic Planning
        pi1_k, m_k = evi(n_states, n_actions, p_hat, confidence_bound_1, r_hat, confidence_bound_2, 1 / np.sqrt(t_k))
        
        #execute policy
        ac = pi1_k[st]
        while(vk[st,ac] != total_numbers[st,ac]):
            next_st, reward, done, info= env.step()
            yield(t, st, ac, next_st, reward)

            #update
            vk[st, ac] =  vk[st, ac] + 1 #update vk(s,a)
            total_rewards[st,ac] += reward # 即時reward
            nk[st, ac] = max(1,  nk[st, ac] + 1) #update nk(s,a)
            t += 1 
            st = next_st
            ac = pi1_k[st]
            total_numbers[st, ac, next_st] += 1 #update nk(s, a, s')








'''if __name__ == '__main__':
    eps = 0.1  # epsilon
    alpha = 0.1  # learning rate
    '''