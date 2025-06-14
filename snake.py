import pygame
import numpy as np
import random
import time
import os
from collections import defaultdict

# Constants for grid and screen dimensions
GRID_SIZE = 20
GRID_WIDTH = 15
GRID_HEIGHT = 15
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE

# Colors for rendering
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
GRAY = (40, 40, 40)

# Q-learning parameters
LEARNING_RATE = 0.05  # Rate at which the agent learns
DISCOUNT = 0.95  # Discount factor for future rewards
EXPLORATION_RATE = 1.0  # Initial exploration rate
EXPLORATION_DECAY = 0.9995  # Decay rate for exploration
MIN_EXPLORATION = 0.01  # Minimum exploration rate

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake AI - Improved")
clock = pygame.time.Clock()

class SnakeGame:
    def __init__(self):
        """
        Initialize the Snake game.
        """
        self.reset()
        
    def reset(self):
        """
        Reset the game state to start a new game.
        """
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Snake starts at the center
        self.direction = (0, 1)  # Initial direction is moving down
        self.food = self._place_food()  # Place the first food
        self.score = 0  # Reset score
        self.game_over = False  # Game is not over
        self.steps_since_food = 0  # Steps taken since last food was eaten
        
    def _place_food(self):
        """
        Place food at a random position on the grid, avoiding the snake's body.
        """
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food
        
    def get_state(self):
        """
        Get the current state of the game for the Q-learning agent.
        The state includes:
        - Danger in all four directions (up, right, down, left)
        - Direction of the food relative to the snake's head
        - Current direction of the snake
        """
        head_x, head_y = self.snake[0]
        
        # Detect danger in all four directions
        danger = [0] * 4
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        
        for i, (dx, dy) in enumerate(directions):
            nx, ny = head_x + dx, head_y + dy
            danger[i] = int(
                nx < 0 or nx >= GRID_WIDTH or
                ny < 0 or ny >= GRID_HEIGHT or
                (nx, ny) in self.snake
            )

        # Determine the direction of the food
        food_dir = [
            int(self.food[1] < head_y),  # Food is above
            int(self.food[1] > head_y),  # Food is below
            int(self.food[0] < head_x),  # Food is to the left
            int(self.food[0] > head_x)   # Food is to the right
        ]

        # Encode the current direction of the snake
        current_dir = [
            int(self.direction == (0, -1)),  # Moving up
            int(self.direction == (0, 1)),   # Moving down
            int(self.direction == (-1, 0)),  # Moving left
            int(self.direction == (1, 0))    # Moving right
        ]

        return tuple(danger + food_dir + current_dir)

    def step(self, action):
        """
        Perform a step in the game based on the given action.
        Actions:
        - 0: Continue in the current direction
        - 1: Turn right
        - 2: Turn left
        Returns:
        - reward: Reward for the action
        - done: Whether the game is over
        """
        # Update direction based on action
        if action == 1:  # Turn right
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # Turn left
            self.direction = (-self.direction[1], self.direction[0])

        # Calculate the new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check for collisions with walls or the snake's body
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or 
            new_head in self.snake):
            self.game_over = True
            return -100, True  # Strong penalty for dying

        # Calculate distance to food for reward adjustment
        old_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        # Update the snake's position
        self.snake.insert(0, new_head)
        self.steps_since_food += 1

        # Check if the snake eats the food
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.steps_since_food = 0
            reward = 20  # Reward for eating food
        else:
            self.snake.pop()  # Remove the tail if no food is eaten
            # Reward based on distance to food
            if new_distance < old_distance:
                reward = 1  # Small reward for getting closer
            else:
                reward = -1  # Small penalty for moving away
        
        # Penalize the snake for taking too long to find food
        if self.steps_since_food > 100:
            reward -= 1

        return reward, False

    def render(self):
        """
        Render the game on the screen.
        """
        screen.fill(BLACK)
        
        # Draw the grid
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # Draw the snake
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else BLUE  # Head is green, body is blue
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw the food
        pygame.draw.rect(screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        
        # Draw the score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.update()

class ImprovedQLearningAgent:
    def __init__(self):
        """
        Initialize the Q-learning agent with a Q-table.
        """
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.learning_rate = LEARNING_RATE
        
    def save_q_table(self, path="improved_q_table.npy"):
        """
        Save the Q-table to a file.
        """
        np.save(path, dict(self.q_table))
        print(f"Q-table saved with {len(self.q_table)} states")

    def load_q_table(self, path="improved_q_table.npy"):
        """
        Load the Q-table from a file if it exists.
        """
        if os.path.exists(path):
            self.q_table = defaultdict(lambda: np.zeros(3), np.load(path, allow_pickle=True).item())
            print(f"Q-table loaded with {len(self.q_table)} states")

    def choose_action(self, state, exploration_rate):
        """
        Choose an action based on the current state and exploration rate.
        """
        if random.random() < exploration_rate:
            return random.randint(0, 2)  # Explore: choose a random action
        return np.argmax(self.q_table[state])  # Exploit: choose the best action

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Q-learning formula.
        """
        current_q = self.q_table[state][action]
        
        if done:
            target = reward  # No future reward if the game is over
        else:
            max_future_q = np.max(self.q_table[next_state])
            target = reward + DISCOUNT * max_future_q
        
        # Standard Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)

# Training function
def train_agent():
    """
    Train the Q-learning agent by playing multiple episodes of the game.
    """
    game = SnakeGame()
    agent = ImprovedQLearningAgent()
    agent.load_q_table()
    
    episodes = 30000  # Number of training episodes
    exploration_rate = EXPLORATION_RATE
    scores = []
    
    print("Training agent...")
    
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0

        while not game.game_over:
            action = agent.choose_action(state, exploration_rate)
            reward, done = game.step(action)
            next_state = game.get_state() if not done else None
            
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        scores.append(game.score)
        exploration_rate = max(MIN_EXPLORATION, exploration_rate * EXPLORATION_DECAY)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            max_score = max(scores) if scores else 0
            print(f"Episode {episode} — Avg Score: {avg_score:.2f} — Max Score: {max_score} — Exploration: {exploration_rate:.3f}")
        
        # Save progress
        if episode % 1000 == 0 and episode > 0:
            agent.save_q_table()

    agent.save_q_table()
    
    # Print final stats
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    final_max = max(scores) if scores else 0
    print(f"\nTraining completed!")
    print(f"Final average score (last 100): {final_avg:.2f}")
    print(f"Maximum score achieved: {final_max}")

# Play function
def play_game():
    """
    Play the game using the trained Q-learning agent.
    """
    game = SnakeGame()
    agent = ImprovedQLearningAgent()
    agent.load_q_table()
    
    print("Playing with trained agent...")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        if not game.game_over:
            state = game.get_state()
            action = agent.choose_action(state, 0)  # No exploration during play
            reward, done = game.step(action)
        
        game.render()
        clock.tick(10)
        
        if game.game_over:
            print(f"Game Over. Score: {game.score}")
            time.sleep(1)
            game.reset()

# Main execution
if __name__ == "__main__":
    """
    Main entry point of the program.
    Check if a trained model exists and either train or play the game.
    """
    if not os.path.exists("improved_q_table.npy"):
        print("No trained model found. Training new agent...")
        train_agent()
    else:
        print("Found existing model. Press 't' to train more, or any other key to play...")
        choice = input().lower()
        if choice == 't':
            train_agent()
    
    play_game()