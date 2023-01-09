import tensorflow as tf
import numpy as np
import random
import cv2
import os
import csv

batch_size = 10
no_of_actions = 10
save_path = 'files1/training/model_files1/cp-{}.ckpt'
save_dir = os.path.dirname(save_path) # 'files/training/model_files'

log_save_path = os.path.join("files1","training","my_logs1")
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)
tf_writer = tf.summary.create_file_writer("files1/training/my_logs1/tf_board")
generations = 10
class DeepQNetwork:

    def __init__(self, training_mode):
        self.epsilon, self.min_epsilon = 0.9, 0.1
        self.decay = self.epsilon/((generations//2)-1)
        self.tranining_mode = training_mode
        self.predict_network = self.build_network()
        self.train_network = self.build_network()
        self.counter = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
        #self.saver = tf.train.Saver(filename = save_path)
        
    # def preprocess(state):
    #     processed_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    #     processed_state = np.reshape(processed_state, newshape=(processed_state.shape, 1))
    #     return processed_state
    def record_summary(self, _loss, counter):
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=_loss)])
        self.writer.add_summary(summary, counter)

    def save_log(self, step, quantity, filename):
        with open(os.path.join(log_save_path, filename), 'a+') as fi:
            csv_w = csv.writer(fi, delimiter=',')
            csv_w.writerow([step, quantity])

    # def save_model(self):
    #     self.saver.save(self.sess, os.path.join(*[".","checkpoints","rl_weights"]),global_step=100)

    # def load_model(self):
    #     save_dir = "model_weights.h5"
    #     latest_weights = tf.train.latest_checkpoint(save_dir)
    #     print("lastest weights", latest_weights)
    #     self.predict_network.load_weights(latest_weights)

    def update_prediction_network(self):
        for train_grad, pred_grad in zip(self.train_network.trainable_variables, self.train_network.trainable_variables):
            pred_grad.assign(train_grad)
        self.train_network.save_weights(save_path.format(self.counter))
        print("leveling up")


    def get_action(self, processed_state):
        
        if np.random.random() > self.epsilon:
            _action = self.get_prediction(np.expand_dims(processed_state, 0))
            
            _action = _action.numpy()
            action = np.argmax(_action)

            print("not random",action)
        else: 
            action = np.random.randint(0, no_of_actions)
            
        return action

    def get_prediction(self, states):
        states = np.reshape(states, newshape=(states.shape[0], 72, 96, 1))/255
        prediction = self.predict_network(states)
        # print("get predictions")
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print(prediction.eval(session=sess))
        # sess.close()
        return prediction
    
    def predict(self, states):
        states = np.reshape(states, newshape=(states.shape[0], 72, 96, 1))/255
        prediction = self.train_network(states)
        # print("train prediction")
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print(prediction.eval(session=sess))
        # sess.close()
        return prediction
    
    def loss(self, ground_truth, prediction):
        loss = tf.keras.losses.mean_squared_error(ground_truth, prediction)
        return loss

    def update_q_value(self, rewards, current_q_list, next_q_list, actions, done, discount_factor):
   
        current_action_qs = current_q_list.numpy()
        next_action_qs = next_q_list.numpy()

        next_max_qs = np.max(next_action_qs, axis = 1)
        new_qs = rewards + (np.ones(done.shape) - done) * discount_factor * next_max_qs
        for i in range(len(current_action_qs)):
            current_action_qs[i, actions[i]] = new_qs[i]
        return current_action_qs

    def train_step(self, states, actions):
        with tf.GradientTape() as tape:
            predictions = self.train_network(states)
            loss = self.loss(actions, predictions)
        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, 10) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        self.train_network.save_weights(save_dir)
        return loss

    def build_network(self):
        inp = tf.keras.layers.Input((72, 96, 1))
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inp)
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(no_of_actions, activation='linear')(x)

        model = tf.keras.Model(inputs=inp, outputs=x)
        model.summary()
        return model


    def train(self, previous_memories):
        self.counter += 1
        current_nodes, actions, next_nodes, rewards, done = previous_memories
        #print(current_nodes.shape, actions.shape, rewards.shape, next_nodes.shape)
        
        current_action_qs = self.predict(current_nodes)
        next_action_qs = self.get_prediction(next_nodes)
        
        current_action_qs = self.update_q_value(rewards, current_action_qs, next_action_qs, actions, done, 0.05)
        current_nodes = np.reshape(current_nodes, newshape=(batch_size, 72,96, 1))/255

        loss = self.train_step(current_nodes, current_action_qs)

        with tf_writer.as_default():
            tf.summary.scalar("loss", data = np.mean(loss), step=self.counter)

        loss = loss.numpy()
        
        self.save_log(self.counter, np.mean(loss), "loss.csv")
