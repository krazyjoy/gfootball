import gfootball.env as football_env
import pandas as pd
import numpy as np
# env = football_env.create_environment(env_name='5_vs_5', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=5 )
# state = env.reset()
# obs, rew, done, info = env.step(env.action_space.sample())
def simplify(obs):
    '''
    input: obs - from 'obs, rew, done, info = env.step(env.action_space.sample())' & chop - 切的多細 e.g. 0.01(粗)->0.0001(細)
    output: dictionary {
        ball: [x,y]
        left_team: [[x1,y1], [x2,y2], ...]
    }
    '''
    # bin = np.arange(-1.5, 1.5, chop)
    # state = {}
    # ball = pd.cut(obs[0]["ball"][:-1],bin)
    # state['ball'] = [ball[0].left, ball[1].left]
    # print(state['ball'])
    # x = []
    # for player in obs[0]["left_team"]:
    #     print("origin", player)
    #     player = pd.cut(player,bin)
    #     x.append([player[0].left, player[1].left])
    #     print("after simplify", player[0].left, player[1].left)
    # state['left_team'] = x
    # print("[state]", state)
    # return state
    observed_states = ["ball_owned_team","ball","left_team","right_team"]
    #print("ball owned")
    #print(obs[0]["ball_owned_team"])
    #print("ball", obs[0]["ball"])
    #print("left team",obs[0]["left_team"])
    states = {state: obs[0][state] if obs[0][state] is not None else None for state in observed_states}
    print("===== states", states)
    bins_width = np.round(np.arange(-1.2,1,0.05),3)
    bins_height = np.round(np.arange(-0.5,0.7,0.025),3)
    bin_states = {}
    for state, value in states.items():
        if state == 'left_team' or state == 'right_team':
            value = value[0]
        if type(value) == int:
            bin_states[state] = [value,-1]
            #print("int",state)
        else:
            print("value[0] state width: ", value[0])
            print("value[1] state height: ", value[1])
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
                    bin_state_h = bins_height[0]
                    break
                elif value[1] < bins_height[bin_h] and value[1] >= bins_height[bin_h-1]:
                    bin_state_h = bins_height[bin_h-1]
                    print("h: bin_state",bin_state_h)
                    break
                elif value[1] > bins_height[len(bins_height)-1]:
                    bin_state_h = bins_height[len(bins_height)-1]
                    break
            print("bin_state_w, bin_state_h", (bin_state_w, bin_state_h))
            bin_states[state] = [bin_state_w, bin_state_h]
    print("bin states", bin_states)
    bin_current_state = [bin_states[s] for s in bin_states]
    print("bin current state", bin_current_state)
    return bin_current_state

# ans = simplify(obs, 0.0005)
# print(ans)