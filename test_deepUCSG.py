from env.SaveExperience import ExperienceHistory
import gfootball.env as football_env
import tensorflow as tf
import tf_slim as slim
import cv2
import numpy as np
import re
import os
import random 
from gfootball.env import football_action_set
import matplotlib.pyplot as plt
def create_network(input, trainable):
    input = input.astype("float16")
    # input: (batch_size, num_frame_stack, 72, 96)
    input_t = tf.expand_dims(input, axis = 0)
    #input_t = input
    if trainable:
        wr = slim.l2_regularizer(1e-6)
    else:
        wr = None

    # 將channel 放在最後一行
    #input_t = tf.transpose(input, [0,2,3,1]) # (batch_size, num_frames, pic_size[0], pic_size[1])


    net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC",
        activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
    tf.print("tensor: ",net)
    net = slim.max_pool2d(net, 2, 2)

    net = slim.conv2d(net, 16, (3, 3), data_format="NHWC",
        activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)

    net = slim.max_pool2d(net, 2, 2)

    net = slim.flatten(net)

    net = slim.fully_connected(net, 1, activation_fn=tf.nn.relu,
        weights_regularizer=wr, trainable=trainable)
    
    tf.print("tensor: ", net)
    return net
    

def extract_feature(model,frame):
    model.fit(frame)
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
                model.output,  # < last layer output 
                model.layers[0].output # < your convolution layer output 
        ]
    )
    return feature_extractor

def UCSG(states, actions, T, delta):
        delta_1 = delta/(2*len(states)*len(states)*len(actions)*np.log2(T))
        p_hat = total_numbers/np.clip(nk.reshape(len(states), len(actions), 1), 1, None)  #拆解
        print("p hat ", p_hat.shape)
        r_hat = total_rewards / np.clip(vk, 1, None)
        upper_conf = 1
        lower_conf = 0
        confidence_bound_1 = np.sqrt((2*len(states)*np.log(1/delta)/len(states)))+ p_hat #(500,19,500)
        print("confidence 1", confidence_bound_1.shape)
        confidence_bound_2_2 = np.clip(np.sqrt(np.log(6/delta_1)/(2*len(states)))+p_hat, None, np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/len(states)) + 7*np.log(6/delta_1)/(3*(len(states)-1))+p_hat)
        print("confidence bound 2", confidence_bound_2_2.shape) #(500,19,500)
        a = np.sqrt(2*np.log(6/delta_1)/(len(states)-1), dtype = np.half)
        b = np.sqrt(p_hat*(1-p_hat), dtype = np.half)
        c1 = (a+b)**2
        c2 = (a-b)**2
        xr = np.clip((1+np.sqrt(1-4*c1))/2, (1-np.sqrt(1-4*c1))/2, 1)
        xl = np.clip((1+np.sqrt(1-4*c2))/2, 0 ,(1-np.sqrt(1-4*c2))/2)
        confidence_bound_2_1 = np.clip(xr, 0 ,xl)
        tmp = confidence_bound_1 + confidence_bound_2_2
        # print('confidencebound1', confidencse_bound_1)
        print("tmp", tmp.shape)
        # print('confidencebound2', confidence_bound_2_2)
        #confidencebound = np.array([ (tmp[i]) for i in range(0, len(tmp)) if tmp[i] not in tmp[:i] ]) # (1,len(action_set),n_states)
        #print("confidencebond", confidence_bound.shape)
        return tmp

def maxminevi(states, actions, gamma, alpha, I, total_rewards, Pk): # I是總共有I個vi
    
    if random.random() < 0.15:
        action_i = random.randint(0,9)
        print("random action: ",actions[action_i])
        return actions[action_i]
    
    v = np.zeros((I, len(states)), dtype = np.int64) # 計算v0 # (100,500)
    print("v shape", v.shape)
    print("pk: ",Pk.shape)
    print("states", states.shape)
    # 計算v1

    ratio = 0.05
    for s in range(len(states)):
        Max = -1
        for a in range(len(actions)):
            Sum = sum(Pk[:][s][a])
            if Sum >= Max:
                Max = Sum
                Max_a_1 = a
        val = total_rewards[s, Max_a_1] + Max
        v[1][s] = ((1-alpha) * val + alpha * v[0][s]) * ratio
    print("v[1][s]",v[1][s])    

    i = 1
    print((max(v[i]) - min(v[i-1])) - (min(v[i]) - max(v[i-1]))) # 0
    while ((max(v[i]) - min(v[i-1])) - (min(v[i]) - max(v[i-1]))) <= (1 - alpha) * gamma: 
        i += 1
        if i == I:
            i -= 1
            break
         # 選一個Pk(s,a)讓值最大。Pk(s,a)是一個一維array
        Max_in = -1
        for s in range(len(states)):
            for a in range(len(actions)):
                Sum = 0
                for next_s in range(len(states)):
                    Sum += Pk[s][a][next_s] * v[i-1][next_s]
                if Sum >= Max_in:
                    Max_in = Sum
                    Max_in_Pk_s = s
                    Max_in_Pk_a = a
        # 每個state都選一個action讓值最大
        for s in range(len(states)):
            Max_out = -1
            for a in range(len(actions)):
                val = total_rewards[s, a] + Max_in
               
                if val >= Max_out:
                    Max_out = val
                    if st_i == s:
                        Max_out_a = a
                        choose_action = Max_out_a
            # print("i-1",i-1)
            # print("v[i-1][s]",v[i-1][s])
            v[i][s] = ((1-alpha) * val + alpha * v[i-1][s]) * ratio

    
       
    print("v[i][s]",v[i][:5])
    try:
        return actions[choose_action]
    except: # 一開始可能不會進到前面的while迴圈，所以直接回傳random的action
        print('!')
        action_i = random.randint(0, len(actions)-1)
        return actions[action_i]




if __name__ == "__main__":

    # set actions
    actions = football_action_set.get_action_set({'action_set': 'default'})
    actions_del = [18, 16, 15, 14, 13, 11, 10, 9, 0]
    for idx in actions_del:
        actions.pop(idx)
    # num_players = 1
    # actions = np.zeros((len(action_set), num_players))

    # all_actions = np.reshape(actions, -1)

    accu_reward = 0
    draw_reward = []
    draw_step = []
    draw_step_ten = []
    draw_reward_ten = []

    states = np.zeros((100)) # 0~500.0

    vk = np.zeros((len(states), len(actions)),dtype = np.int64) #vk(s,a)
    total_numbers = np.zeros((len(states), len(actions), len(states)), dtype = np.int64) # nk(s,a,s')
    total_rewards = np.zeros((len(states), len(actions)), dtype = np.float16) #maximin
    nk = np.ones((len(states), len(actions)), dtype = np.int64) #nk(s,a)



    env = football_env.create_environment(env_name="1_vs_1_easy", representation='pixels', render = True, 
            rewards="easy,scoring", number_of_left_players_agent_controls = 1
        )
    frame = env.reset()

    steps = 0
    
    ac_i = random.randint(0,9)
    print("first action: ",actions[ac_i])
   
    frame, reward, done, info = env.step(ac_i)
    net = create_network(frame, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("net: ",net.eval(session = sess))
        #st_i = np.round(net.eval(),1)
        st_i = net.eval(session = sess).astype(int)
        print("st_i: ",st_i)
        if st_i[0][0] > 99:
            st_i[0][0] = 99
        sess.close()
    all_tensors = st_i
    confidence_bound = UCSG(states, actions, 1e3, 0.01)


    while steps < 600:
        

        
        steps += 1
        print("steps: ",steps)

        ac = maxminevi(states, actions, 0.01, 0.9, 20, total_rewards, confidence_bound)
        next_frame, reward, done, info = env.step(ac)
        print("reward", reward)
        print("action",ac)

        
        next_net = create_network(next_frame,True)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("next_net",next_net.eval(session = sess))
            #next_st_i = np.round(next_net.eval(),1)
            next_st_i = next_net.eval(session = sess).astype(int)
            print("next_st_i: ",next_st_i)
            if next_st_i[0][0] > 99:
                next_st_i[0][0] = 99
            
            sess.close()

        all_tensors = tf.concat([all_tensors, next_st_i], axis = 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("all_tensors",all_tensors.eval(session = sess))


        
        

        # states, actions, gamma, alpha, I, total_rewards, Pk
        

        for a in range(len(actions)):
            if ac == actions[a]:
                ac_i = a
        vk[st_i, ac_i] =  vk[st_i, ac_i] + 1 #update vk(s,a)

        total_rewards[st_i,ac_i] += reward.astype(float) # 即時reward
        nk[st_i, ac_i] = max(1,  nk[st_i, ac_i] + 1) #update nk(s,a)
        total_numbers[st_i, ac_i, next_st_i] += 1 #update nk(s, a, s')
       
        next_st_i = st_i

        accu_reward += reward
        print("accu_reward",accu_reward)
        if(steps % 100 == 0):
            print('step:', steps)
            draw_step.append(steps)
            draw_reward.append(accu_reward)
        if(steps % 10 == 0):
            draw_step_ten.append(steps)
            draw_reward_ten.append(accu_reward)

    print('accumulate reward:', accu_reward)

    print("steps", draw_step)
    plt.plot(draw_step,draw_reward)
    plt.title("UCSG Reward Convergence Rate") # title
    plt.xlabel("Steps") # y label
    plt.ylabel("Reward") # x label
    plt.show()

    plt.plot(draw_step_ten,draw_reward_ten)
    plt.title("UCSG Reward Convergence Rate") # title
    plt.xlabel("Steps") # y label
    plt.ylabel("Reward") # x label
    plt.show()
        # print("vk(s,a): ")
        # print(vk.nonzero()) # (array([0], dtype=int64), array([0], dtype=int64))

        # print("total_rewards(s,a): ")
        # print(total_rewards.nonzero())

        # print("nk(s,a): ")
        # print(nk.nonzero())

        # print("total_numbers(s,a,s'): ") # (array([  0,   0,   0, ..., 499, 499, 499], dtype=int64), array([ 0,  1,  2, ..., 16, 17, 18], dtype=int64))
        # print(total_numbers.nonzero())



