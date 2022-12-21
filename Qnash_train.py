import gfootball.env as football_env
import numpy as np
import random
import math
import nashpy as nash
import matplotlib.pyplot as plt
from Env_discrete import simplify
import copy

nactions = 8
discount = 1
alpha = 0.4
epsilon =  0.01

def flat_states(obs):
    """
    ball: shape(1,3)
    ball_owned_team: (int)
    ball_owned_player: (int) 
    left_team: shape(nagents,2)
    left_team_role: shape(1,nagents) x
    score: shape(1,2) x
    steps_left: (int) x
    """
    minor_states = ['ball','ball_owned_team','ball_owned_player',
                        'left_team','left_team_roles','score','steps_left']

    
    states = {state: obs[0][state] if obs[0][state] is not None else None for state in minor_states}
 

    print(states)
    new_states = []
    for state in states.values():
        if type(state) == np.ndarray:
          state = list(state.flatten())[0]
          new_states.append(state)
        elif type(state) == list:
          for s in range(len(state)):
            new_states.append(state[s])
        else:
          new_states.append(state)
    print(new_states)


def bin_state(obs):
    observed_states = ["ball_owned_team","ball","left_team","right_team"]
    print("ball owned")
    print(obs[0]["ball_owned_team"])
    print("ball", obs[0]["ball"])
    print("left team",obs[0]["left_team"])
    states = {state: obs[0][state] if obs[0][state] is not None else None for state in observed_states}
    print("states", states)
    bins_width = np.round(np.arange(-1.2,1,0.05),3)
    bins_height = np.round(np.arange(-0.5,0.7,0.025),3)
    bin_states = {}
    for state, value in states.items():
        if state == 'left_team' or state == 'right_team':
            value = value[0]
        if type(value) == int:
            bin_states[state] = [value,-1]
            print("int",state)
        else:
            print("not int",state)
            bin_states[state] = []
            for bin_w in range(1,len(bins_width)):
                if bins_width[0] > value[0]:
                    print("width out of bound",value[0])
                    bin_state_w = bins_width[0]
                    break
                if value[0] < bins_width[bin_w] and value[0] >= bins_width[bin_w-1]:
                    bin_state_w = bins_width[bin_w-1]
                    print("w: bin_state",bin_state_w)
                    break
                
            for bin_h in range(1,len(bins_height)):
                if bins_height[0] > value[1]:
                    print("height out of bound",value[1])
                    bin_state_h = bins_height[1]
                    bin_states[state] = [bin_state_w, bin_state_h]
                    break
                if value[1] < bins_height[bin_h] and value[1] >= bins_height[bin_h-1]:
                    bin_state_h = bins_height[bin_h-1]
                    print("h: bin_state",bin_state_h)
                    break
            bin_states[state] = [bin_state_w, bin_state_h]
                    

                
    bin_current_state = [bin_states[s] for s in bin_states]
    print("bin current state", bin_current_state)
    return bin_current_state

def create_q_tables():
    bin_width = 0.2
    bin_height = 0.1
    ball_own = 3 # -1,0,1
    width = 2 # 10 states [-1,1]
    height = 1 # 5 states [-.42, 0.42]
    W_bin = np.round(np.arange(-1.2,1.2,0.05), 3)
    print(W_bin)
    H_bin = np.round(np.arange(-0.5,0.6,0.025),3)
    states_table = {}
    states_table["ball_owned"] = np.array([[-1,-1],[0,-1],[1,-1]])
    states_table["ball"] = []
    states_table["left_team"] = []
    states_table["right_team"] = []
    states_table["action1"] = []
    states_table["action2"] = []
    for w in W_bin:
        for h in H_bin:
            #print("w,h",(w,h))
            states_table["ball"].append([w,h])
            states_table["left_team"].append([w,h])
    states_table["ball"] = np.array(states_table["ball"])
    states_table["left_team"] = np.array(states_table["left_team"])
    states_table["right_team"] = copy.deepcopy(states_table["left_team"])
    states_table["action1"] = np.arange(0,8,1)
    states_table["action2"] = np.arange(0,8,1)
    #print(list(states_table.values()))
    #print(states_table["ball_owned"].shape)
    #print(states_table["ball"].shape)
    #print(states_table["left_team"].shape)
    qtables1 = np.zeros((3,
    W_bin.shape[0]*H_bin.shape[0],
    W_bin.shape[0]*H_bin.shape[0],
    nactions, nactions))
    
    qtables2 = np.zeros((3,
    W_bin.shape[0]*H_bin.shape[0],
    W_bin.shape[0]*H_bin.shape[0],
    nactions, nactions))
    
    

    return states_table, qtables1, qtables2

def find_states(states_table, states, team):
    """
    find state:
    "ball_owned": [[-1,-1],[0, -1], [1,-1]]
    "ball": [[-1, -0.8, -0.6,...., 0.8,1] ,[-0.5, -0.4, ... , 0.4, 0.5]] # 11 x 11
    "left_team":    
    "right_team":
    bins_width = np.round(np.arange(-1,1,0.2),3)
    bins_height = np.round(np.arange(-0.5,0.5,0.1),3)
    """

    find = []
    

    for i, key in enumerate(list(states_table.keys())[:4]):
        if team == 1 and i == 3:
            # left team
            break
        if team == 2 and i == 2:
            continue
        for j, s in enumerate(states_table[key]):
            #print("s",s)
            #print("states i ",states[i])
            if tuple(states[i]) == tuple(s):
                find.append([i,j])
                # print("f:",find)
                continue
    # print("find states",find)
    return find

def GetPi(qtables1, qtables2, find1, find2):
    print("get pi find", find1[1][1])
    print("get pi find", find1[2][1])
    print("get pi find", find2[1][1])
    print("get pi find", find2[2][1])
    Pi = []
    Pi_O = []
    for i in range(nactions):
        row_q = []
        row_opponent = []
        for j in range(nactions):
            row_q.append(qtables1[find1[0][1], find1[1][1], find1[2][1], i,j])
            row_opponent.append(qtables2[find2[0][1], find2[1][1], find2[2][1], i,j])

        Pi.append(row_q)
        Pi_O.append(row_opponent)
    # print("PI", Pi)
    # print("Pi_O", Pi_O)  
    nash_game = nash.Game(Pi,Pi_O)
    equilibria = nash_game.lemke_howson_enumeration()
    pi_nash = None
    try:
        pi_nash_list = list(equilibria)
       
    except:
        pi_nash_list = []
    for index, eq in enumerate(pi_nash_list):
        if eq[0].shape == (nactions, ) and eq[1].shape == (nactions, ):
            if any(np.isnan(eq[0])) == False and any(np.isnan(eq[1])) == False:
                if index != 0:
                    pi_nash = (eq[0], eq[1])
                    break
    if pi_nash is None:
        print("pi_nash is null, bug in nashpy")
        pi_nash = (np.ones(nactions)/nactions, np.ones(nactions)/nactions)
    return pi_nash[0], pi_nash[1]

def computeNashQ(qtable1, qtable2, agent, find1, find2):
    Pi, Pi_O = GetPi(qtable1, qtable2, find1, find2)
    nashq = 0

    for action1 in range(nactions):
        for action2 in range(nactions):
            if agent == 1:
                nashq += Pi[action1] * Pi_O[action2] * qtable1[find1[0][1], find1[1][1], find1[2][1], action1, action2]
            elif agent == 2:
                nashq += Pi[action1] * Pi_O[action2] * qtable2[find2[0][1], find2[1][1], find2[2][1], action1, action2]
            else:
                print("error agent")
    print("agent: ",agent)
    print("nashq: ",nashq)
    return nashq

def computeQ(agent, qtable1, qtable2, find1, find2, rewards, action1, action2):
    nashq = computeNashQ(qtable1, qtable2,agent,find1, find2)
    if agent == 1:
         m_value = alpha * (rewards + discount * nashq)
         print("m_val", m_value)
         o_value = (1-alpha) * qtable1[find1[0][1], find1[1][1], find1[2][1],action1, action2]
         print("o_val",o_value)
         qtable1[find1[0][1], find1[1][1], find1[2][1], action1, action2] = o_value + m_value
         print("qtable1 change: ",qtable1[find1[0][1], find1[1][1], find1[2][1], action1, action2])
         return qtable1
    else:
        m_value = alpha * ((-rewards) + discount * nashq)
        o_value = (1-alpha) * qtable2[find2[0][1], find2[1][1], find2[2][1],action1, action2]
        qtable2[find2[0][1], find2[1][1], find2[2][1], action1, action2] = o_value + m_value
        print("qtable2 change: ",qtable2[find2[0][1], find2[1][1], find2[2][1], action1, action2])
        return qtable2

def choose_action(qtable1, qtable2, states_table, states, epsilon = 0.01):
    print(states)
    find1 = find_states(states_table, states, 1)
    find2 = find_states(states_table, states, 2)

    Pi, Pi_O = GetPi(qtable1, qtable2, find1, find2)
    #print("pi shape ",Pi.shape)
    

    if random.random() <= epsilon:
        print("random_action")
        action_idx = random.randint(0,nactions-1)
        print(action_idx)
    else:
        print("best action")
        #print("Pi",Pi)
        action_idx = random.choice(np.flatnonzero(Pi == Pi.max()))
        print(action_idx)

    if random.random() <= epsilon:
        print("random_action")
        opponent_action_idx = random.randint(0,nactions-1)
        print(opponent_action_idx)
    else:
        print("best action")
        #print("Pi",Pi_O)
        opponent_action_idx = random.choice(np.flatnonzero(Pi_O == Pi_O.max()))
        print(opponent_action_idx)
    return action_idx, opponent_action_idx


#%%
print("start")
env = football_env.create_environment(env_name='1_vs_1_easy', representation='raw', render='True', number_of_left_players_agent_controls=1 , rewards = "easy,scoring")
obs = env.reset() # states_of_agent_i = obs[agent_i] (type dic)
#states = flat_states(obs)
states_table, qtable1, qtable2 = create_q_tables()
# bin_states = bin_state(obs)
bin_states = simplify(obs)

steps = 0
accu_reward = 0
draw_reward = []
draw_step = []
loop = 3
while loop > 0:
    if steps == 300:
        loop -= 1
        break
    #find = find_states(states_table,bin_states,1)
    action1,action2 = choose_action(qtable1, qtable2, states_table, bin_states, epsilon)

    print("action1: {}; action2: {}", action1, action2)
    obs, reward, done, info = env.step(action1)
    print("while reward: ", reward)
    if done:
        break
    accu_reward += reward
    print("acc_reward",accu_reward)
    #states = flat_states(obs)
    # bin_states = bin_state(obs)
    bin_states = simplify(obs)

    find1 = find_states(states_table, bin_states, team = 1)
    find2 = find_states(states_table, bin_states, team = 2)
    print("find1", find1)
    print("find2", find2)
    # update
    qtable1 = computeQ(1, qtable1, qtable2, find1, find2, reward,action1,action2) 
    qtable2 = computeQ(2, qtable2, qtable2, find1, find2, reward,action1,action2)
    #print("qtable1",qtable1)
    #print("qtable2",qtable2)
    steps += 1
    if(steps%100==0):
        draw_step.append(steps)
        draw_reward.append(accu_reward)

print("steps", draw_step)
plt.plot(draw_step,draw_reward)
plt.title("Qnash Reward Convergence Rate") # title
plt.xlabel("Steps") # y label
plt.ylabel("Reward") # x label
plt.show()
