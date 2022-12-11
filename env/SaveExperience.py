import numpy as np

class ExperienceHistory:

    def __init__(self, num_frame_stack = 4,
                capacity=int(1e5), pic_size=(72,96)
    ):
        self.num_frame_stack = num_frame_stack # state is stack of num_frame_stack frames
        self.capacity = capacity # I in UCSG
        self.pic_size = pic_size
        self.counter = 0
        self.frame_window = None
        self.init_caches() # create cache variables
        self.expecting_new_episode = True

    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype="float32")
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype = "int16"
        )
        self.next_states = -np.ones((self.capacity, self.num_frame_stack),
            dtype="int16"
        )
        self.is_done = -np.ones(self.capacity, "int16")
        self.actions = -np.ones(self.capacity, dtype="int16")

        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = -np.ones((self.max_frame_cache, ) + self.pic_size, dtype = "float16") # (max_frame_cache, 72, 96)

    

    def start_new_episode(self, frame):
        """
        padding all memory cache
        """
        assert self.expecting_new_episode, "previous episode didn't end yet"

        frame_idx = self.counter % self.max_frame_cache

        # 取 num_frame_stack 張 圖片, 利用frame_idx 取得stack中圖片
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)  #np.repeat(1,5) => [1,1,1,1,1]
        # memory space
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def current_state(self):
        assert self.frame_window is not None, "do something first"
        # print(self.frames[self.frame_window].shape) (4,72,96)
        return self.frames[self.frame_window]

    def add_experience(self, frame, action, done, reward):
        assert self.frame_window is not None, "start episode first"

        # 取得新的 window index
        self.counter += 1

        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        
        # 更新window
        self.frame_window = np.append(self.frame_window[1:], frame_idx) # 每一次更新都捨棄window 第一個元素

        self.next_states[exp_idx] = self.frame_window 
        
        self.actions[exp_idx] = action

        self.is_done[exp_idx] = done

        self.frames[frame_idx] = frame

        self.rewards[exp_idx] = reward

        if done:
            self.expecting_new_episode = True

    def sample_mini_batch(self, n_samples):
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size = n_samples)

        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]

        return {
            "reward": self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]

        }






