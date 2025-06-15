# Snake AI - Q-Learning Implementation

This project demonstrates a Snake game with an AI, trained with Q-learning. The AI learns to play the game during training. Receives rewards for eating food, and penalties for collisions or inefficient moves.

## Features

- **AI Agent**: Trained using Q-learning to play Snake game.
- **Training Mode**: Trains the AI if no Q-table exists.
- **Pygame**: Visualizes the game.
- **Improved Reward System**: Rewards for getting closer to food and penalties for moving away or taking too long.
- **Q-Table Persistence**: Saves and loads the Q-table to/from a file.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   Ensure Python is installed, then run:

   ```bash
   pip install pygame numpy
   ```

3. Run the project:
   ```bash
   python snake.py
   ```

## How It Works

### Q-Learning Algorithm

- **State Representation**:
  The state is represented in a tuple containing:

  - Danger indicators for the four directions (up, down, left, right).
  - Relative position of the food.
  - Current movement direction.

- **Action Space**:
  The AI can choose one of the three:

  - Continue moving straight.
  - Turn left.
  - Turn right.

- **Reward System**:

  - +20 for eating food.
  - -100 for collisions.
  - +1 for moving closer to food.
  - -1 for moving away from food.
  - Additional penalties for taking too long to find food.

- **Q-Table**:
  The Q-table is a dictionary where keys are states, and values are arrays representing the Q-values for each action. For example:
  ```python
  q_table = {
      state_1: [q_value_straight, q_value_left, q_value_right],
      state_2: [q_value_straight, q_value_left, q_value_right],
  }
  ```

### Training

If the `improved_q_table.npy` file does not exist, the AI will start training for n episodes. Training progress is displayed in the console.

- **Training Loop**:
  The training loop iterates over episodes, where the AI plays the game, updates the Q-table using the Bellman equation, and adjusts its strategy based on exploration (epsilon-greedy policy).

  ```python
  for episode in range(num_episodes):
      state = reset_environment()
      while not game_over:
          action = select_action(state, epsilon)
          next_state, reward = perform_action(action)
          update_q_table(state, action, reward, next_state)
          state = next_state
  ```

- **Bellman Equation**:
  The Q-value updates like this:
  ```python
  q_table[state][action] = q_table[state][action] + learning_rate * (
      reward + discount_factor * max(q_table[next_state]) - q_table[state][action]
  )
  ```

### Gameplay

Once trained, the AI plays the game automatically.

### Game Environment

The game environment is implemented using `pygame`. It provides:

- **Grid-based Representation**:
  The Snake and food are represented on a grid.

- **Collision Detection**:
  The environment checks for collisions with walls or the Snake's body.

- **Reward Assignment**:
  Rewards are given depending on the AI's actions, such as eating food, moving closer to food, or colliding.

## Files

- **`snake.py`**: Contains the implementation of the game and AI agent.
- **`improved_q_table.npy`**: Stores the Q-table for the trained AI. If not present, the AI will train itself.
