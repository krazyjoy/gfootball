from SaveExperience import ExperienceHistory
import gfootball.env as football_env
import tensorflow as tf
import tf_slim as slim
import cv2
import numpy as np
import re
import os

class NN:
    def __init__(self, env, batch_size = 2, pic_size=(72,96,3), num_frame_stack = 1,
                gamma = 0.95, action_map = None, optimizer_params = None, 
                network_update_freq = 100,

    ):
        self.env = env

        self.exp_history = ExperienceHistory(num_frame_stack = num_frame_stack, capacity = int(10), pic_size = pic_size)

        self.playing_cache = ExperienceHistory(num_frame_stack=num_frame_stack,capacity = num_frame_stack * 5 + 10, pic_size =pic_size)

        if action_map is not None:
            self.dim_actions = len(action_map)
        else:
            self.dim_actions = 19

        self.num_frame_stack = num_frame_stack

        self.pic_size = pic_size

        self.gamma = gamma

        self.regularization = 1e-6
        self.batch_size = batch_size

        self.optimizer_params = optimizer_params or dict(learning_rate = 0.0004, epsilon = 1e-7)

        self.do_training = True

        self.session = None

        self.network_update_freq = network_update_freq

        self.action_map = action_map

        self.global_counter = 0

        self.episode_counter = 0

        self.actions = np.zeros((5,19))
        self.states = np.zeros((500)) # 0~500.00

        self.vk = np.zeros((len(self.states), len(self.actions))) #vk(s,a)
        self.total_numbers = np.zeros((len(self.states), len(self.actions), len(self.states))) # nk(s,a,s')
        self.total_rewards = np.zeros((len(self.states), len(self.actions))) #maximin
        self.nk = np.ones((len(self.states), len(self.actions))) #nk(s,a)

    def AllStates():
        state_list = []

    def play_episode(self):
        # train mode or play mode: store history
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        frames_in_episode = 0

        # env reset:
        first_frame = self.env.reset() # (72,96,3)
        #first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) # (72,96)
        
        # create a memory space
        eh.start_new_episode(first_frame)
        first_frame = np.expand_dims(first_frame, axis = 0)
        print("first frame expands: ", first_frame.shape)



        nn_state = self.create_network(tf.convert_to_tensor(first_frame, dtype = "float16"), trainable = True)
        
        nn_state = tf.reshape(nn_state, (nn_state.shape[1],))
        print("nn_state.shape",nn_state.shape)

        
        delta = 0.01
        T = 2000
        while True:
            if self.global_counter == 2000:
                break
            if np.random.rand() > 0.1:
                action_idx = self.session.run(
                    self.best_action,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            else:
                action_idx = np.random.choice(19)
            
            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx
            
            reward = 0


            

            # skip frames = 1
            for _ in range(1):
                observation, r, done, info = env.step(action)
                reward += r
                if done:
                    break
            
            total_reward += reward
            frames_in_episode += 1

            observation = np.expand_dims(observation, axis = 0)
            eh.add_experience(observation, action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq:
                    # 需用到session, 將當下fixed params 指派給 target params
                    self.update_target_network()

                # training
                train_cond = (
                    self.exp_history.counter >= 1000 and 
                    self.global_counter % 4 == 0

                )
                if train_cond:
                    

                    """ calculate confidence bound """
                    delta_1 = delta/(2*len(self.states)*len(self.states)*len(self.actions)*np.log2(T))
                    p_hat = self.total_numbers/np.clip(self.nk.reshape(len(self.states), len(self.actions), 1), 1, None)  #拆解
                    r_hat = self.total_rewards / np.clip(self.vk, 1, None)
                    upper_conf = 1
                    lower_conf = 0

                    # Update the confidence set
                    #confidence_bound_1 = np.clip(np.sqrt((2*n_states*np.log(1/delta)/n_states))+ p_hat, 0, 1)         
                    confidence_bound_1 = np.sqrt((2*len(self.states)*np.log(1/delta)/len(self.states)))+ p_hat
                    confidence_bound_2_2 = np.clip(np.sqrt(np.log(6/delta_1)/(2*len(self.states)))+p_hat, None, np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/len(self.states)) + 7*np.log(6/delta_1)/(3*(len(self.states)-1))+p_hat)
                    #confidence_bound_2_2 = np.clip(confidence_bound_2_2, 0, 1)
                    a = np.sqrt(2*np.log(6/delta_1)/(len(self.states)-1))
                    b = np.sqrt(p_hat*(1-p_hat))
                    c1 = (a+b)**2
                    c2 = (a-b)**2
                    xr = np.clip((1+np.sqrt(1-4*c1))/2, (1-np.sqrt(1-4*c1))/2, 1)
                    xl = np.clip((1+np.sqrt(1-4*c2))/2, 0 ,(1-np.sqrt(1-4*c2))/2)
                    confidence_bound_2_1 = np.clip(xr, 0 ,xl)
                    tmp = confidence_bound_1 + confidence_bound_2_2
                    # print('confidencebound1', confidence_bound_1)
                    # print('confidencebound2', confidence_bound_2_2)
                    confidencebound = [ (tmp[i]) for i in range(0, len(tmp)) if tmp[i] not in tmp[:i] ]



                    """ calculate confidence bound """

                    self.train() # calculate loss, update optimizer

                
                if done:
                    if self.do_training:
                        self.episode_counter += 1
                
                    return total_reward, frames_in_episode




    
    def build_graph(self):
        """ init inputs for conv2d """
        # (batch_size, 4, 72, 96)
        input_dim_with_batch = (self.batch_size, ) + self.pic_size

        input_dim_general = (None, ) + self.pic_size #　(None, 4, 72, 96)

        self.input_prev_state = tf.placeholder(tf.float16, input_dim_general, "prev_state")

        self.input_next_state = tf.placeholder(tf.float16, input_dim_with_batch, "next_state")

        self.input_reward = tf.placeholder(tf.float16, self.batch_size, "reward")

        self.input_actions = tf.placeholder(tf.int32, self.batch_size, "actions")

        self.input_done_mask = tf.placeholder(tf.int32, self.batch_size, "done_mask")

        
        # self.input_state = tf.placeholder(tf.float16, (self.batch_size, ) + self.pic_size, "input_state")

        # self.input_next_state = tf.placeholder(tf.float16,(self.batch_size, ) + self.pic_size, "next_input")
        # all_states = tf.placeholder
        """ state action values: Vi """
        # # V[I][s]

        # with tf.variable_scope("fixed"):
        #     nn_next_state = self.create_network(self.input_next_state, trainable = False)

        with tf.variable_scope("train"):
            nn_prev_state = self.create_network(self.input_prev_state, trainable = True)
            
        # """ best actions: maxminevi: greatest vi """
        self.best_action = tf.convert_to_tensor(np.repeat(3,self.batch_size))

        not_done = tf.cast(tf.logical_not(tf.cast(self.input_done_mask, "bool")), "float16")

        """ calculate loss """

        """ optimizer """

        # q_target = tf.reduce_max(qsa_targets, -1) * self.gamma * not_done + self.input_reward
        # # # select the chosen action from each row
        # # # in numpy this is qsa_estimates[range(batchsize), self.input_actions]
        # action_slice = tf.stack([tf.range(0, self.batch_size), self.input_actions], axis=1)
        # q_estimates_for_input_action = tf.gather_nd(qsa_estimates, action_slice)

        # training_loss = tf.nn.l2_loss(q_target - q_estimates_for_input_action) / self.batch_size

        # optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))

        # reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        # self.train_op = optimizer.minimize(reg_loss + training_loss)

        train_params = self.get_variables("train")
        # fixed_params = self.get_variables("fixed")


        # # assert (len(train_params) == len(fixed_params))

        # 記住一個batch的output
        # self.copy_network_ops = [tf.assign(fixed_v, train_v)
        #     for train_v, fixed_v in zip(train_params, fixed_params)]
 
    
    def get_variables(self, scope):
        vars = [t for t in tf.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)
    
    
    def create_network(self, input, trainable):
        
        # input: (batch_size, num_frame_stack, 72, 96)
        #input_t = tf.expand_dims(input, axis = -1)
        input_t = input
        if trainable:
            wr = slim.l2_regularizer(self.regularization)
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

        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)

        # q_state_action_values = slim.fully_connected(net, self.dim_actions,
        #     activation_fn=None, weights_regularizer=wr, trainable=trainable)

        # print("q state action vals: ", q_state_action_values)

        return net

    def update_target_network(self):
        # 將fixed_params指派給target_params
        #self.session.run(self.copy_network_ops)
        return

    def train(self):
        """ input : state, action, state """
        batch = self.exp_history.sample_mini_batch(self.batch_size)

        fd = {
            self.input_reward: "reward",
            self.input_prev_state:"prev_state",
            self.input_next_state:"next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
       
        # frame dictionary:
        #  key: "reward", "prev_state", "next_state", "actions", "done_mask" 
        #  value: batch["reward"], ...
        fd1 = {ph:batch[k] for ph,k in fd.items()}

        """ CALCULATE PK """
        # vk:
        # total_numbers:
        # total_rewards:
        # s
        # update optimizer, with optimizer.minimize(reg_loss + training_loss)
        # self.session.run([self.train_op], fd1) 
        """ output: VI """

        #action = maxminevi()
        #return action

# to start training from scratch:
load_checkpoint = False
checkpoint_path = "data/checkpoint03"
train_episodes = 25
save_freq_episodes = 1

def one_episode(agent):
    reward, frames = agent.play_episode()
    print("episode: %d, reward: %f, length: %d, total steps: %d" %
        (agent.episode_counter, reward, frames, agent.global_counter))
    
    save_cond = (
        agent.episode_counter % save_freq_episodes == 0
        and checkpoint_path is not None
        and agent.do_training
    )
    if save_cond:
        print("save checkpoint")
        save_checkpoint(agent)


def save_checkpoint(agent):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    p = os.path.join(checkpoint_path, "m.ckpt")
    saver.save(sess, p, agent.global_counter)
    print("save to %s - %d" %(p, agent.global_counter))

if __name__ == "__main__":

    
    env = football_env.create_environment(env_name="1_vs_1_easy", representation='pixels', render = True, number_of_left_players_agent_controls=1,
        rewards="checkpoints,scoring"
    )

    agent = NN(env=env)
    agent.build_graph() # tensorflow init
    sess = tf.InteractiveSession()
    agent.session = sess

    saver = tf.train.Saver(max_to_keep = 100)
    if load_checkpoint:
        # load model
        print("path: %s" % checkpoint_path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        assert ckpt, "checkpoint path %s not found" %checkpoint_path
        # get counter
        global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
        saver.restore(sess, ckpt.model_checkpoint_path)
        agent.global_counter = global_counter
    else:
        if checkpoint_path is not None:
            assert not os.path.exists(checkpoint_path), \
                "checkpoint path already exists but load_checkpoint is false"
            
            tf.global_variables_initializer().run()




    tf.global_variables_initializer().run()

    while True:
        if agent.do_training and agent.episode_counter > train_episodes:
            break
        one_episode(agent)


