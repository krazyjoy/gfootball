from collections import deque
import gfootball.env as football_env
from deepnetwork import DeepQNetwork
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
        #self.dqn.load_model()
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

        for episode in tqdm(range(episodes)):

            
            
            self.observation = self.env.reset()
            print("obs: ",self.observation.shape)

            
            step_counter = 0

            # action set 


            print("actions length", actions)
            # random first action
            
            action_i = random.randint(0, len(actions)-1)
        
            print("actions_i", action_i)
            print("(actions[action_i]",self.actions[action_i])
            # print("first action: ",actions[ac_i])
            frame, reward, done, info = self.env.step(actions[action_i])

            reward_history = [reward]
            processed_state = self.preprocess(frame)


            print("p state", processed_state.dtype)

            step_history = [step_counter]
            while step_counter < 1000:
                action = self.dqn.get_action(processed_state)

                self.next_observation, reward, done, info = self.env.step(action)
                
                next_processed_state = self.preprocess(self.next_observation)

                
                reward_history.append(reward - reward_history[-1])
                
                
                #self.previous_memory.append([self.observation, action, self.next_observation, reward, 1 if done else 0])

                self.previous_memory.append([processed_state, action, next_processed_state, reward, 1 if done else 0])

                self.observation = self.next_observation

                step_counter += 1

                step_history.append(step_counter)
                # training after one batch
                if step_counter%10 == 0:
                    if len(self.previous_memory) >= batch_size:
                        self.train_network()
                if done:
                    break
                with tf_writer.as_default():
                    tf.summary.scalar("Episodic Average Rewards", np.mean(reward_history))
                    tf.summary.scalar("Epsilon", self.dqn.epsilon)
                self.dqn.save_log(episode, np.mean(reward_history), "episodic_reward.csv")

                if generations//2 >= episode >=1:
                    new_epsilon = self.dqn.epsilon-self.dqn.decay
                    self.dqn.epsilon = max(new_epsilon, self.dqn.min_epsilon)
                if episode>9 and episode%10 == 0:
                    self.dqn.update_prediction_network()
            return step_history, reward_history

save_path = 'files/training/model_files/cp-{}.ckpt'
save_dir = os.path.dirname(save_path)
log_save_path = os.path.join("files","training","my_logs")
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)

tf_writer = tf.contrib.summary.create_file_writer("files/training/my_logs/tf_board")

actions = football_action_set.get_action_set({'action_set': 'default'})
actions_del = [18, 16, 15, 14, 13, 11, 10, 9, 0]
for idx in actions_del:
    actions.pop(idx)
env = football_env.create_environment(env_name="1_vs_1_easy", representation='pixels', render = True,   
rewards="easy,scoring", number_of_left_players_agent_controls = 1)             
agent = Agent(env, actions)
generations = 20
step_counter = []
acc_rewards = []
acc_rewards.extend(agent.run(episodes = generations)) 
plt.plot(step_counter, acc_rewards)
plt.title("DQN")
plt.xlabel("steps")
plt.ylabel("Rewards")