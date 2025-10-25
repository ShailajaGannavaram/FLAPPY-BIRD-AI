# flappy_hybrid.py
# Hybrid: Behavior Cloning (from your demos) + Double DQN (fine-tune)
# Save/load model: dqn_hybrid.pth

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time

# -------------------------
# Config
# -------------------------
MODEL_FILE = "dqn_hybrid.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------
# Environment
# -------------------------
class FlappyBirdEnv:
    def __init__(self):
        # sizes & bird initial params
        self.width = 288
        self.height = 512
        self.bird_x = 50
        self.BIRD_SIZE = 20

        # pipe & game parameters (must exist before reset)
        self.pipe_width = 50
        self.pipe_gap = 100
        self.pipe_velocity = 3

        # physics defaults (potentially adjusted for manual)
        self.gravity = 0.5
        self.flap_strength = -8.0

        # create initial state via reset
        self.reset()

    def reset(self):
        self.bird_y = self.height // 2
        self.bird_vel = 0.0
        self.pipe_x = self.width + random.randint(0, 50)
        self.pipe_height = random.randint(120, 320)
        self.score = 0
        self.done = False
        self.prev_center_dist = abs(self.bird_y - (self.pipe_height + self.pipe_gap / 2.0))
        return self.get_state()

    def get_state(self):
        return np.array([
            self.bird_y / self.height,
            self.bird_vel / 10.0,
            (self.pipe_x - self.bird_x) / self.width,
            self.pipe_height / self.height
        ], dtype=np.float32)

    def step(self, action):
        # action: 0 = nothing, 1 = flap
        if action == 1:
            self.bird_vel = self.flap_strength
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel
        self.pipe_x -= self.pipe_velocity

        # base survival reward
        reward = 0.05

        # shaping: reward for getting closer to pipe gap center
        center_y = self.pipe_height + self.pipe_gap / 2.0
        center_dist = abs(self.bird_y - center_y)
        if center_dist < self.prev_center_dist:
            reward += 0.2
        else:
            reward -= 0.03
        self.prev_center_dist = center_dist

        # passed pipe reward
        if (self.pipe_x + self.pipe_width) < self.bird_x and not self.done:
            self.score += 1
            reward += 5.0
            # spawn next pipe
            self.pipe_x = self.width + random.randint(0, 80)
            self.pipe_height = random.randint(120, 320)
            self.prev_center_dist = abs(self.bird_y - (self.pipe_height + self.pipe_gap / 2.0))

        # collisions
        if self.bird_y - self.BIRD_SIZE / 2 <= 0 or self.bird_y + self.BIRD_SIZE / 2 >= self.height:
            self.done = True
            reward = -10.0

        # pipe collision (box)
        bird_top = self.bird_y - self.BIRD_SIZE / 2
        bird_bottom = self.bird_y + self.BIRD_SIZE / 2
        bird_left = self.bird_x - self.BIRD_SIZE / 2
        bird_right = self.bird_x + self.BIRD_SIZE / 2

        pipe_left = self.pipe_x
        pipe_right = self.pipe_x + self.pipe_width
        pipe_top = self.pipe_height
        pipe_bottom = self.pipe_height + self.pipe_gap

        if bird_right > pipe_left and bird_left < pipe_right:
            if bird_top < pipe_top or bird_bottom > pipe_bottom:
                self.done = True
                reward = -10.0

        return self.get_state(), reward, self.done

# -------------------------
# Networks and Buffer
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d
    def __len__(self):
        return len(self.buffer)

# -------------------------
# Agent
# -------------------------
class Agent:
    def __init__(self, training_style=2, model_file=MODEL_FILE):
        self.device = DEVICE
        self.model = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # training style params
        if training_style == 1:  # Pro
            self.memory_capacity = 20000
            self.batch_size = 128
            self.epsilon = 1.0
            self.epsilon_min = 0.03
            self.epsilon_decay = 0.995
            self.train_iters = 3
            self.tau = 0.02
        elif training_style == 3:  # Ultra safe
            self.memory_capacity = 8000
            self.batch_size = 64
            self.epsilon = 1.0
            self.epsilon_min = 0.04
            self.epsilon_decay = 0.997
            self.train_iters = 1
            self.tau = 0.005
        else:  # Champion
            self.memory_capacity = 12000
            self.batch_size = 128
            self.epsilon = 1.0
            self.epsilon_min = 0.035
            self.epsilon_decay = 0.996
            self.train_iters = 2
            self.tau = 0.01

        self.memory = ReplayBuffer(capacity=self.memory_capacity)
        self.gamma = 0.99
        self.action_dim = 2
        self.model_file = model_file
        self.best_score = -999

        # safe load
        if os.path.exists(self.model_file):
            try:
                saved = torch.load(self.model_file, map_location=self.device)
                self.model.load_state_dict(saved)
                self.target.load_state_dict(self.model.state_dict())
                self.epsilon = max(0.1, self.epsilon * 0.2)
                print("Loaded model:", self.model_file)
            except Exception as e:
                print("Saved model incompatible, starting fresh. (", e, ")")

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(s)
        return int(q.argmax().cpu().item())

    def store(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    def soft_update(self):
        for targ, src in zip(self.target.parameters(), self.model.parameters()):
            targ.data.copy_((1 - self.tau) * targ.data + self.tau * src.data)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q_vals = self.model(s).gather(1, a)

        with torch.no_grad():
            next_actions = self.model(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions)
            target_q = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_multiple(self, iters=1):
        for _ in range(iters):
            self.train_step()
        self.soft_update()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        try:
            torch.save(self.model.state_dict(), self.model_file)
            print("Model saved to", self.model_file)
        except Exception as e:
            print("Save failed:", e)

    # Behavior cloning pretrain (supervised) - trains model to imitate actions in demos
    def behavior_clone(self, demo_states, demo_actions, epochs=8, batch_size=64):
        if len(demo_states) == 0:
            print("No demo data for BC.")
            return
        demo_states = torch.FloatTensor(np.array(demo_states)).to(self.device)
        demo_actions = torch.LongTensor(np.array(demo_actions)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        dataset_size = len(demo_states)
        for e in range(epochs):
            perm = np.random.permutation(dataset_size)
            total_loss = 0.0
            for i in range(0, dataset_size, batch_size):
                idx = perm[i:i+batch_size]
                bs = demo_states[idx]
                ba = demo_actions[idx]
                logits = self.model(bs)
                loss = criterion(logits, ba)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bs.size(0)
            avg = total_loss / dataset_size
            print(f"BC Epoch {e+1}/{epochs} loss: {avg:.4f}")
        # update target and reduce epsilon a bit after BC
        self.target.load_state_dict(self.model.state_dict())
        self.epsilon = max(0.1, self.epsilon * 0.2)
        print("Behavior cloning finished. Epsilon set to", self.epsilon)

# -------------------------
# Game + demo recording
# -------------------------
class FlappyGame:
    def __init__(self):
        pygame.init()
        self.width = 288
        self.height = 512
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird - BC + Double DQN Hybrid")
        self.clock = pygame.time.Clock()
        self.env = FlappyBirdEnv()
        self.agent = None
        self.mode = None
        self.training_style = 2

    def choose_training_style(self):
        choosing = True
        font = pygame.font.SysFont(None, 26)
        while choosing:
            self.screen.fill((240,240,240))
            lines = [
                "Choose training style (1/2/3):",
                "1 - Pro Gamer (fast)",
                "2 - Champion (balanced) [default]",
                "3 - Ultra Safe (slower)"
            ]
            y = 110
            for line in lines:
                self.screen.blit(font.render(line, True, (20,20,20)), (16, y))
                y += 32
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.training_style = 1; choosing = False
                    elif event.key == pygame.K_2:
                        self.training_style = 2; choosing = False
                    elif event.key == pygame.K_3:
                        self.training_style = 3; choosing = False

    def select_mode_ui(self):
        selecting = True
        font = pygame.font.SysFont(None, 28)
        while selecting:
            self.screen.fill((255,255,255))
            self.screen.blit(font.render("Press R to Record Demos", True, (0,0,0)), (20,160))
            self.screen.blit(font.render("Press M for Manual Play", True, (0,0,0)), (20,200))
            self.screen.blit(font.render("Press A for AI Play (learn)", True, (0,0,0)), (20,240))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        selecting = False
                        return 'record'
                    if event.key == pygame.K_m:
                        selecting = False
                        return 'manual'
                    if event.key == pygame.K_a:
                        selecting = False
                        return 'ai'

    def draw(self):
        self.screen.fill((180, 220, 255))
        # ground
        pygame.draw.rect(self.screen, (170,110,40), (0, self.height - 30, self.width, 30))
        # bird
        bird_x = self.env.bird_x
        bird_y = int(self.env.bird_y)
        B = self.env.BIRD_SIZE
        pygame.draw.ellipse(self.screen, (255, 60, 60), (bird_x - B//2, bird_y - B//2, B, B))
        # pipes
        pygame.draw.rect(self.screen, (30,160,40), (self.env.pipe_x, 0, self.env.pipe_width, self.env.pipe_height))
        pygame.draw.rect(self.screen, (30,160,40), (self.env.pipe_x, self.env.pipe_height + self.env.pipe_gap, self.env.pipe_width, self.height))
        # overlays
        font = pygame.font.SysFont(None, 22)
        self.screen.blit(font.render(f"Score: {self.env.score}", True, (10,10,10)), (8, 8))
        pygame.display.flip()

    def record_demos(self, num_episodes=10, max_steps=800):
        print("Recording demos. Press SPACE to flap. Recording episodes:", num_episodes)
        demos_states = []
        demos_actions = []
        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); return demos_states, demos_actions
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            action = 1
                next_state, reward, done = self.env.step(action)
                demos_states.append(state.copy())
                demos_actions.append(action)
                state = next_state
                steps += 1
                self.draw()
                self.clock.tick(60)
            print(f"Recorded demo ep {ep} with score {self.env.score}")
            pygame.time.delay(500)
        print("Demo recording complete. Total transitions:", len(demos_states))
        return demos_states, demos_actions

    def run(self):
        # choose style
        self.choose_training_style()
        # UI select mode or record
        choice = self.select_mode_ui()

        # init agent
        self.agent = Agent(training_style=self.training_style, model_file=MODEL_FILE)

        # if record requested:
        if choice == 'record':
            # ask number of demos in terminal
            try:
                n = int(input("How many demo episodes to record? (recommended 8-12): ") or "10")
            except:
                n = 10
            demos_s, demos_a = self.record_demos(num_episodes=n)
            # quick behavior cloning
            print("Starting Behavior Cloning pretrain from demos...")
            self.agent.behavior_clone(demos_s, demos_a, epochs=6, batch_size=64)
            print("BC pretrain finished. Now choose mode to continue.")
            # after BC, let user choose manual or ai
            choice = self.select_mode_ui()

        # set mode
        if choice == 'manual':
            self.mode = 'manual'
            self.env.gravity = 0.3
            self.env.flap_strength = -7.0
        else:
            self.mode = 'ai'

        # training / play loop
        episodes = 200  # default; you can change in code or later prompt
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); return
                    if self.mode == 'manual' and event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            action = 1
                if self.mode == 'ai':
                    action = self.agent.select_action(state)

                next_state, reward, done = self.env.step(action)

                # store and train
                # always allow learn-from-manual (so even manual mode improves agent)
                self.agent.store(state, action, reward, next_state, float(done))
                if self.mode == 'ai':
                    self.agent.train_multiple(iters=self.agent.train_iters)
                else:
                    # when manual, do light training to learn quickly from you
                    self.agent.train_multiple(iters=1)

                state = next_state
                ep_reward += reward
                steps += 1
                self.draw()
                self.clock.tick(60)

            # episode ended - bookkeeping
            # soft update more
            self.agent.soft_update()

            # save if improved
            if self.env.score > self.agent.best_score:
                self.agent.best_score = self.env.score
                self.agent.save()

            # periodic autosave
            if ep % 30 == 0:
                self.agent.save()

            # report
            print("=" * 40)
            print(f"Episode {ep}/{episodes} ended. Mode: {self.mode.upper()} Score: {self.env.score} Steps: {steps} EpReward: {ep_reward:.2f}")
            print(f"Epsilon: {self.agent.epsilon:.4f} Best: {self.agent.best_score}")
            print("Next game starting in 1s...")
            print("=" * 40)
            pygame.time.delay(800)

        # final save
        self.agent.save()
        pygame.quit()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    game = FlappyGame()
    game.run()
