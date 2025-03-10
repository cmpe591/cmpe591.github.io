# Using multiple processes to collect data for training a model

import time
from collections import deque

import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import numpy as np

import environment


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._delta = 0.05

        self._goal_thresh = 0.01
        self._max_timesteps = 50

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100*np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100*np.linalg.norm(obj_pos - goal_pos), 1)
        return 1/(ee_to_obj) + 1/(obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        action = action.clamp(-1, 1).cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


class Memory:
    def __init__(self, keys, buffer_length=None):
        self.buffer = {}
        self.keys = keys
        for key in keys:
            self.buffer[key] = deque(maxlen=buffer_length)

    def clear(self):
        for key in self.keys:
            self.buffer[key].clear()

    def append(self, dic):
        for key in self.keys:
            self.buffer[key].append(dic[key])

    def sample_n(self, n):
        r = torch.randperm(len(self))
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        res = {}
        for key in self.keys:
            res[key] = torch.stack([self.buffer[key][i] for i in idx])
        return res

    def get_all(self):
        idx = list(range(len(self)))
        return self.get_by_idx(idx)

    def __len__(self):
        return len(self.buffer[self.keys[0]])


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Linear(128, 2)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.linear(x)
        return x


def collecter(model, shared_queue, is_collecting, is_finished, device):
    env = Hw3Env(render_mode="offscreen")
    while not is_finished.is_set():
        while is_collecting.is_set():
            env.reset()
            state = env.state()
            done = False
            cum_reward = 0.0
            while not done:
                with torch.no_grad():
                    action = model(state.to(device))
                next_state, reward, is_terminal, is_truncated = env.step(action[0])
                cum_reward += reward
                done = is_terminal or is_truncated
                shared_queue.put(((state*255).byte(), action, reward, (next_state*255).byte(), done))
                state = next_state
                if is_finished.is_set():
                    break
            if is_finished.is_set():
                break
        if is_finished.is_set():
            break
        is_collecting.wait()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    device = torch.device("cpu")
    model = MyModel().to(device)
    replay_buffer = Memory(["state", "action", "reward", "next_state", "done"], buffer_length=10000)
    model.share_memory()  # share model parameters across processes
    shared_queue = mp.Queue()
    is_collecting = mp.Event()
    is_finished = mp.Event()

    is_collecting.set()
    procs = []
    for i in range(6):
        p = mp.Process(target=collecter, args=(model, shared_queue, is_collecting, is_finished, device))
        p.start()
        procs.append(p)

    num_updates = 10
    for i in range(num_updates):
        start = time.time()
        buffer_feeded = 0
        while buffer_feeded < 400:
            print(f"Buffer size={len(replay_buffer)}", end="\r")
            if not shared_queue.empty():
                # unfortunately, you can't feed the replay buffer as fast as you collect
                state, action, reward, next_state, done = shared_queue.get()
                replay_buffer.append({"state": state.clone(),
                                      "action": action.clone(),
                                      "reward": reward,
                                      "next_state": next_state.clone(),
                                      "done": done})
                del state, action, reward, next_state, done
                buffer_feeded += 1
        end = time.time()
        is_collecting.clear()
        print(f"{400/(end-start):.2f} samples/sec... Updating model...")
        # do your update
        is_collecting.set()
    is_finished.set()
    for p in procs:
        p.join()
