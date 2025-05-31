import pygame
import numpy as np
import random
import time
import os
from collections import defaultdict

# Constants
GRID_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
GRAY = (40, 40, 40)

# Q-learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.999
MIN_EXPLORATION = 0.01

# Pygame init
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake AI")
clock = pygame.time.Clock()

class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (0, 1)
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        
    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food
        
    def get_state(self):
        head_x, head_y = self.snake[0]
        danger = [0] * 4
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (dx, dy) in enumerate(directions):
            nx, ny = head_x + dx, head_y + dy
            danger[i] = int(
                nx < 0 or nx >= GRID_WIDTH or
                ny < 0 or ny >= GRID_HEIGHT or
                (nx, ny) in self.snake
            )

        food_dir = [
            int(self.food[1] < head_y),
            int(self.food[1] > head_y),
            int(self.food[0] < head_x),
            int(self.food[0] > head_x)
        ]

        current_dir = [
            int(self.direction == (0, -1)),
            int(self.direction == (0, 1)),
            int(self.direction == (-1, 0)),
            int(self.direction == (1, 0))
        ]

        return tuple(danger + food_dir + current_dir)

    def step(self, action):
        if action == 1:
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:
            self.direction = (-self.direction[1], self.direction[0])

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or 
            new_head in self.snake[:-1]):
            self.game_over = True
            return -100, True  # strong penalty for dying

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.pop()
            reward = 0  # no penalty/reward for normal move

        return reward, False

    def render(self):
        screen.fill(BLACK)
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else BLUE
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        pygame.display.update()

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(3))

    def save_q_table(self, path="q_table.npy"):
        np.save(path, dict(self.q_table))

    def load_q_table(self, path="q_table.npy"):
        if os.path.exists(path):
            self.q_table = defaultdict(lambda: np.zeros(3), np.load(path, allow_pickle=True).item())
            print("Q-table loaded.")

    def choose_action(self, state, exploration_rate):
        if random.random() < exploration_rate:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        self.q_table[state][action] = new_q

# Game and agent
game = SnakeGame()
agent = QLearningAgent()
agent.load_q_table()

# Training loop
if not os.path.exists("q_table.npy"):
    episodes = 7000
    exploration_rate = EXPLORATION_RATE
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0

        while not game.game_over:
            action = agent.choose_action(state, exploration_rate)
            reward, done = game.step(action)
            next_state = game.get_state()
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        exploration_rate = max(MIN_EXPLORATION, exploration_rate * EXPLORATION_DECAY)
        if episode % 50 == 0:
            print(f"Episode {episode} — Score: {game.score} — Total Reward: {total_reward} — Exploration: {exploration_rate:.3f}")

    agent.save_q_table()

# Play loop
game.reset()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    state = game.get_state()
    action = agent.choose_action(state, exploration_rate=0)
    _, done = game.step(action)
    game.render()
    clock.tick(10)
    if done:
        print(f"Game Over. Score: {game.score}")
        time.sleep(1)
        game.reset()
