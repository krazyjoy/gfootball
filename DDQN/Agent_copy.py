from collections import deque
import gfootball.env as football_env
from deepnetwork_copy import DeepQNetwork
import random
from gfootball.env import football_action_set
from tqdm import tqdm
import numpy as np
import cv2
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
batch_size = 10
class Agent:
    def __init__(self, env, actions):
        self.dqn = DeepQNetwork(training_mode = True)
        # self.dqn.load_model()
        self.dqn.epsilon = 0.1
        self.node_history_size = 200
        self.previous_memory = deque(maxlen=self.node_history_size)
        self.env = env
        self.actions = actions
    
    def preprocess(self,state):
        processed_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        processed_state = np.reshape(processed_state, newshape=(processed_state.shape[0],processed_state.shape[1], 1))
        processed_state = processed_state.astype("float32")
        return processed_state

    def get_batch(self, sampling_size):
        this_batch = random.sample(self.previous_memory, sampling_size)
        current_nodes, actions, next_nodes, rewards, done = list(zip(*this_batch))
        return [np.stack(current_nodes), np.array(actions), np.stack(next_nodes), np.array(rewards), np.array(done)]

    def train_network(self):
        data = self.get_batch(batch_size)
        self.dqn.train(data)
    def run(self, episodes, train_frequency = 2):

        total_reward_history = []
        reset_reward_history = []
        step_history = []
        episodes = 20
        no_reset_total_reward = 0
        self.observation = self.env.reset()
        for episode in tqdm(range(episodes)):

            step_counter = 1
            total_reward = 0

            
            
            print("obs: ",self.observation.shape)
            print("actions length", actions)
            # random first action
            
            action_i = random.randint(0, len(actions)-1)
        
            print("actions_i", action_i)
            print("(actions[action_i]",self.actions[action_i])
            # print("first action: ",actions[ac_i])
            frame, reward, done, info = self.env.step(self.actions[action_i])



            processed_state = self.preprocess(frame)
            
            step_history.append(step_counter + episode * 10000)

            reset_reward_history.append(total_reward)
            
            total_reward_history.append(no_reset_total_reward)
            
            while step_counter < 10001:

                if episode == 0:
                    action_i = random.randint(0, len(self.actions)-1)
        
                    print("actions_i", action_i)
                    print("(actions[action_i]",self.actions[action_i])

                # else:
                action_i = self.dqn.get_action(processed_state)
                print("action dqn", self.actions[action_i])

                
                self.next_observation, reward, done, info = self.env.step(self.actions[action_i])
                print("action", self.actions[action_i])
                next_processed_state = self.preprocess(self.next_observation)

                print("step_counter ",step_counter)
                no_reset_total_reward += reward
                print("no reset total reward ",no_reset_total_reward)
                total_reward += reward
                print("total reward ",total_reward)
                
                #print("reward_history", reward_history)
                
                #self.previous_memory.append([self.observation, action, self.next_observation, reward, 1 if done else 0])
                
                flag = False
                flag = np.array_equal(processed_state,next_processed_state)
                if flag == True:  
                    print("wrong state")
                    break
                self.previous_memory.append([processed_state, action_i, next_processed_state, reward, 1 if done else 0])
                
                processed_state = next_processed_state

                self.observation = self.next_observation

                step_counter += 1

                # if (step_counter% 20 == 0):
                total_reward_history.append(no_reset_total_reward)
                reset_reward_history.append(total_reward)
                step_history.append(step_counter + episode * 10000)

                #step_history.append(step_counter*(episode+1))
                # training after one batch
                if step_counter%10 == 0:
                    if len(self.previous_memory) >= batch_size:
                        self.train_network()
                if done:
                    break
                with tf_writer.as_default():
                    tf.summary.scalar("Episodic Average Rewards", data=np.mean(reset_reward_history), step=episode)
                    tf.summary.scalar("Epsilon", data=self.dqn.epsilon, step=episode)
                self.dqn.save_log(episode, np.mean(reset_reward_history), "episodic_reward.csv")

                if generations//2 >= episode >=1:
                    new_epsilon = self.dqn.epsilon-self.dqn.decay
                    self.dqn.epsilon = max(new_epsilon, self.dqn.min_epsilon)
                if (step_counter % 500 == 0 and episode == 0) or (step_counter % 800 == 0 and episode > 0):
                    self.dqn.update_prediction_network()
            
        plt.plot(step_history, total_reward_history)
        
        plt.title("DQN")
        plt.xlabel("steps")
        plt.ylabel("total Rewards")  
        plt.show()
        plt.plot(step_history, reset_reward_history)   
        plt.title("DQN")
        plt.xlabel("steps")
        plt.ylabel("episode Rewards")  
        plt.show()   
        return step_history, reset_reward_history

save_path = 'files1/training/model_files1/cp-{}.ckpt'
save_dir = os.path.dirname(save_path)
log_save_path = os.path.join("files","training","my_logs2")
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)

tf_writer = tf.summary.create_file_writer("files1/training/my_logs1/tf_board")

actions = football_action_set.get_action_set({'action_set': 'default'})
actions_del = [18, 16, 15, 14, 13, 11, 10, 9, 0]
for idx in actions_del:
    actions.pop(idx)
env = football_env.create_environment(env_name="1_vs_1_easy", representation='pixels', render = True,   
rewards="checkpoints,scoring", number_of_left_players_agent_controls = 1)             

# no_reset = 0
# no_reset_history = []
# reset_history = []
# step_history = []
# for episode in range(3):

#     total_reward = 0

#     steps = 1
#     while steps < 21:
#         reward = 3

#         total_reward += reward

#         no_reset += reward

#         steps += 1

#         step_history.append(20*episode + steps )
#         no_reset_history.append(total_reward)

#         reset_history.append(no_reset)

# plt.plot(step_history, no_reset_history)
# plt.show()
# plt.plot(step_history,reset_history)
# plt.show()
agent = Agent(env, actions)
generations = 200
step_counter = []
acc_rewards = []
step_history, reward_history = agent.run(episodes = generations)
difference_reward = [reward_history[0]]
for i in range(1,len(reward_history)):
    difference_reward.append(reward_history[i] - reward_history[i-1])

plt.plot(step_history, difference_reward)
plt.title("DQN Reward Difference Reward") # title
plt.xlabel("Steps") # y label
plt.ylabel("Difference Reward") # x label
plt.show()
