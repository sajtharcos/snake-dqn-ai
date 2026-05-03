import pygame
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


WIDTH = 700
HEIGHT = 500
GAME_WIDTH = 500
BLOCK = 20

FPS_HUMAN = 10
FPS_AI = 60

MODE = "ai"
# MODE = "human"


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 240
        self.y = 240
        self.direction = (BLOCK, 0)

        self.snake = [
            (self.x, self.y),
            (self.x - BLOCK, self.y),
            (self.x - 2 * BLOCK, self.y)
        ]

        self.score = 0
        self.frame = 0
        self.game_over = False
        self.spawn_food()
        return self.get_state()

    def spawn_food(self):
        while True:
            fx = random.randrange(0, GAME_WIDTH, BLOCK)
            fy = random.randrange(0, HEIGHT, BLOCK)
            self.food = (fx, fy)

            if self.food not in self.snake:
                break

    def collision(self, point):
        x, y = point

        if x < 0 or x >= GAME_WIDTH or y < 0 or y >= HEIGHT:
            return True

        if point in self.snake[1:]:
            return True

        return False

    def human_step(self, new_direction=None):
        if new_direction is not None:
            if new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]:
                self.direction = new_direction

        return self.move()

    def ai_step(self, action):
        directions = [
            (BLOCK, 0),
            (0, BLOCK),
            (-BLOCK, 0),
            (0, -BLOCK)
        ]

        index = directions.index(self.direction)

        if action == 0:
            self.direction = directions[index]
        elif action == 1:
            self.direction = directions[(index + 1) % 4]
        elif action == 2:
            self.direction = directions[(index - 1) % 4]

        return self.move()

    def move(self):
        self.frame += 1

        head_x, head_y = self.snake[0]
        dx, dy = self.direction

        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])

        new_head = (head_x + dx, head_y + dy)

        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_distance < old_distance:
            reward = 0.2
        else:
            reward = -0.2

        if self.collision(new_head):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True, self.score

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()

        if self.frame > 200 * len(self.snake):
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True, self.score

        return self.get_state(), reward, False, self.score

    def get_state(self):
        head = self.snake[0]
        x, y = head

        point_left = (x - BLOCK, y)
        point_right = (x + BLOCK, y)
        point_up = (x, y - BLOCK)
        point_down = (x, y + BLOCK)

        dir_left = self.direction == (-BLOCK, 0)
        dir_right = self.direction == (BLOCK, 0)
        dir_up = self.direction == (0, -BLOCK)
        dir_down = self.direction == (0, BLOCK)

        danger_straight = (
            (dir_right and self.collision(point_right)) or
            (dir_left and self.collision(point_left)) or
            (dir_up and self.collision(point_up)) or
            (dir_down and self.collision(point_down))
        )

        danger_right = (
            (dir_up and self.collision(point_right)) or
            (dir_down and self.collision(point_left)) or
            (dir_left and self.collision(point_up)) or
            (dir_right and self.collision(point_down))
        )

        danger_left = (
            (dir_down and self.collision(point_right)) or
            (dir_up and self.collision(point_left)) or
            (dir_right and self.collision(point_up)) or
            (dir_left and self.collision(point_down))
        )

        food_x, food_y = self.food

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(dir_left),
            int(dir_right),
            int(dir_up),
            int(dir_down),

            int(food_x < x),
            int(food_x > x),
            int(food_y < y),
            int(food_y > y)
        ]

        return np.array(state, dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x, return_activations=False):
        z1 = self.fc1(x)
        a1 = self.relu(z1)

        z2 = self.fc2(a1)
        a2 = self.relu(z2)

        q = self.fc3(a2)

        if return_activations:
            return q, a1, a2

        return q


class Agent:
    def __init__(self):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=100_000)

        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

        self.batch_size = 1000

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_long(self):
        if len(self.memory) < self.batch_size:
            sample = self.memory
        else:
            sample = random.sample(self.memory, self.batch_size)

        if len(sample) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*sample)
        self.train_step(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone().detach()

        for i in range(len(dones)):
            q_new = rewards[i]

            if not dones[i]:
                with torch.no_grad():
                    q_new = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))

            target[i][actions[i]] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()


def draw_neurons(screen, values, x, y, title, font, max_neurons=32):
    text = font.render(title, True, (255, 255, 255))
    screen.blit(text, (x, y - 22))

    values = np.array(values[:max_neurons], dtype=np.float32)

    max_val = max(float(np.max(np.abs(values))), 1e-6)

    for i, v in enumerate(values):
        intensity = int(255 * abs(float(v)) / max_val)
        color = (intensity, intensity, intensity)

        pygame.draw.rect(
            screen,
            color,
            (x + i * 5, y, 4, 28)
        )


def draw_game(screen, game, font):
    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 0, 0), (*game.food, BLOCK, BLOCK))

    for i, part in enumerate(game.snake):
        if i == 0:
            color = (0, 255, 0)
        else:
            color = (0, 180, 0)

        pygame.draw.rect(screen, color, (*part, BLOCK, BLOCK))

    pygame.draw.line(screen, (80, 80, 80), (GAME_WIDTH, 0), (GAME_WIDTH, HEIGHT), 2)

    score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))


def draw_ai_info(screen, font, game, agent, old_state):
    state_tensor = torch.tensor(old_state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_vals, a1, a2 = agent.model(state_tensor, return_activations=True)

    q_vals = q_vals[0].detach().numpy()
    a1 = a1[0].detach().numpy()
    a2 = a2[0].detach().numpy()

    panel_x = 515

    title = font.render("Neural view", True, (255, 255, 0))
    screen.blit(title, (panel_x, 15))

    draw_neurons(screen, old_state, panel_x, 70, "Input", font, max_neurons=11)
    draw_neurons(screen, a1, panel_x, 145, "Hidden 1", font, max_neurons=32)
    draw_neurons(screen, a2, panel_x, 220, "Hidden 2", font, max_neurons=32)
    draw_neurons(screen, q_vals, panel_x, 295, "Output Q", font, max_neurons=3)

    q_text = font.render(
        f"S:{q_vals[0]:.2f} R:{q_vals[1]:.2f} L:{q_vals[2]:.2f}",
        True,
        (255, 255, 255)
    )
    screen.blit(q_text, (panel_x, 340))

    eps_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, (255, 255, 255))
    screen.blit(eps_text, (panel_x, 370))


def play_human():
    pygame.init()

    screen = pygame.display.set_mode((GAME_WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - Human")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    game = SnakeGame()
    running = True
    game_over = False

    while running:
        new_direction = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if game_over and event.key == pygame.K_r:
                    game.reset()
                    game_over = False

                if not game_over:
                    if event.key == pygame.K_w:
                        new_direction = (0, -BLOCK)
                    elif event.key == pygame.K_s:
                        new_direction = (0, BLOCK)
                    elif event.key == pygame.K_a:
                        new_direction = (-BLOCK, 0)
                    elif event.key == pygame.K_d:
                        new_direction = (BLOCK, 0)

        if not game_over:
            _, _, done, _ = game.human_step(new_direction)

            if done:
                game_over = True

        draw_game(screen, game, font)

        if game_over:
            text = font.render("Game Over - R ujra", True, (255, 0, 0))
            screen.blit(text, (160, 240))

        pygame.display.update()
        clock.tick(FPS_HUMAN)

    pygame.quit()


def play_ai():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - AI + neuron visualization")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    game = SnakeGame()
    agent = Agent()

    games_played = 0
    record = 0

    running = True

    while running:
        old_state = game.get_state()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.choose_action(old_state)

        new_state, reward, done, score = game.ai_step(action)

        agent.train_short(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            games_played += 1
            record = max(record, score)

            agent.train_long()

            print(
                f"Game: {games_played} | "
                f"Score: {score} | "
                f"Record: {record} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            game.reset()

        draw_game(screen, game, font)
        draw_ai_info(screen, font, game, agent, old_state)

        info = font.render(
            f"Game: {games_played} | Record: {record}",
            True,
            (255, 255, 255)
        )
        screen.blit(info, (515, 420))

        pygame.display.update()
        clock.tick(FPS_AI)

    pygame.quit()


if __name__ == "__main__":
    if MODE == "human":
        play_human()
    elif MODE == "ai":
        play_ai()