import pygame
import sys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize
pygame.init()

# Screen Setup
WIDTH, HEIGHT = 650, 630
GRID_SIZE = 45
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Drone Navigation")

# Clock
clock = pygame.time.Clock()

# Load Background
sky_bg = pygame.image.load(r"C:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\assets\background.jpg")
sky_bg = pygame.transform.scale(sky_bg, (WIDTH, HEIGHT))

home_bg= pygame.image.load(r"C:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\assets\bg1.png")
home_bg = pygame.transform.scale(home_bg, (WIDTH, HEIGHT))

# Load Drone Image
drone_img = pygame.image.load(r"c:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\assets\drone.png").convert_alpha()
drone_img = pygame.transform.scale(drone_img, (GRID_SIZE, GRID_SIZE))

# Load Goal Image
goal_img = pygame.image.load(r"C:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\assets\goal.png").convert_alpha()
goal_img = pygame.transform.scale(goal_img, (GRID_SIZE, GRID_SIZE))

# Load Obstacle Images
obstacle_imgs = []
for i in range(1, 4):
    img = pygame.image.load(f"C:\\Users\\PAUSHALI MONDAL\\Desktop\\pythonfiles\\REINFORECEMENT LEARNING\\assets\obstacle{i}.png").convert_alpha()
    img = pygame.transform.scale(img, (GRID_SIZE, GRID_SIZE))
    obstacle_imgs.append(img)
# # Icons
# restart_icon = pygame.image.load(r"C:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\restart.png").convert_alpha()
# restart_icon = pygame.transform.scale(restart_icon, (60, 60))

# quit_icon = pygame.image.load(r"C:\Users\PAUSHALI MONDAL\Desktop\pythonfiles\REINFORECEMENT LEARNING\quit.png").convert_alpha()
# quit_icon = pygame.transform.scale(quit_icon, (60, 60))


# Game Elements
drone_start = [1, 1]
goal_pos = [ROWS - 2, COLS - 2]
walls = []
obstacle_objects = []
for _ in range(60):
    pos = [random.randint(0, ROWS - 1), random.randint(0, COLS - 1)]
    if pos != drone_start and pos != goal_pos and pos not in walls:
        walls.append(pos)
        img = random.choice(obstacle_imgs)
        obstacle_objects.append((pos, img))

# RL Parameters
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # L, R, U, D
q_table = np.zeros((ROWS, COLS, len(actions)))
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.999
alpha = 0.1
gamma = 0.9
episodes = 3000

episode_rewards = []
epsilon_values = []
steps_per_episode=[]

# Fonts
font = pygame.font.SysFont("Verdana", 26)

def show_home_screen():
    title_font = pygame.font.SysFont("Verdana", 40, bold=True)
    sub_font = pygame.font.SysFont("Verdana", 24)

    # Display background
    screen.blit(home_bg, (0, 0))

    # Title Text
    title_text = title_font.render("2D Drone Navigation", True, (0 , 0, 0))
    sub_text = sub_font.render("Press any key to start", True, (0, 0, 0))

    # ⬇️ Shifted both lines lower by increasing y values
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
    sub_rect = sub_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 90))

    screen.blit(title_text, title_rect)
    screen.blit(sub_text, sub_rect)
    pygame.display.flip()

    # Wait for key press
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                waiting = False
            



# Drawing function
def draw_game(drone_pos, path=None, steps=None):
    screen.blit(sky_bg, (0, 0))

    # Draw all obstacles
    for pos, img in obstacle_objects:
        x, y = pos[1] * GRID_SIZE, pos[0] * GRID_SIZE
        screen.blit(img, (x, y))

    # Draw goal
    gx, gy = goal_pos[1] * GRID_SIZE, goal_pos[0] * GRID_SIZE
    screen.blit(goal_img, (gx, gy))

    # Draw path as solid black lines
    if path and len(path) > 1:
        points = [(p[1] * GRID_SIZE + GRID_SIZE // 2, p[0] * GRID_SIZE + GRID_SIZE // 2) for p in path]
        pygame.draw.lines(screen, (0, 0, 0), False, points, 3)  # 3 is line thickness

    # Draw drone
    dx, dy = drone_pos[1] * GRID_SIZE, drone_pos[0] * GRID_SIZE
    screen.blit(drone_img, (dx, dy))

    # Draw steps
    if steps is not None:
        step_box = pygame.Rect(WIDTH - 160, 10, 140, 40)
        pygame.draw.rect(screen, (255, 255, 255), step_box)
        step_text = font.render(f"Steps: {steps}", True, (0, 0, 0))
        screen.blit(step_text, (WIDTH - 150, 15))

    pygame.display.flip()




# RL training function
def train_q_learning():
    global epsilon
    for ep in range(episodes):
        state = drone_start[:]
        total_reward = 0
        steps = 0  # Initialize steps counter for each episode
        visited = set()
        for _ in range(500):
            row, col = state
            visited.add(tuple(state))

            # Epsilon-Greedy Policy
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, len(actions) - 1)
            else:
                action_index = np.argmax(q_table[row, col])

            dx, dy = actions[action_index]
            new_row, new_col = row + dy, col + dx

            # Check bounds and walls
            if 0 <= new_row < ROWS and 0 <= new_col < COLS and [new_row, new_col] not in walls:
                if [new_row, new_col] == goal_pos:
                    reward = 100
                elif (new_row, new_col) in visited:
                    reward = -10
                else:
                    reward = -1

                    # 📍 Add distance-based reward shaping here
                    old_distance = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
                    new_distance = abs(new_row - goal_pos[0]) + abs(new_col - goal_pos[1])
                    if new_distance < old_distance:
                        reward += 1  # Encourages progress
                    else:
                        reward -= 1  # Penalizes moving away
            else:
                reward = -100
                new_row, new_col = row, col

            total_reward += reward
            steps += 1  # Increment steps for each move

            # Q-Learning Update
            old_q = q_table[row, col, action_index]
            future_q = np.max(q_table[new_row, new_col])
            q_table[row, col, action_index] = old_q + alpha * (reward + gamma * future_q - old_q)

            state = [new_row, new_col]
            if state == goal_pos:
                break

        episode_rewards.append(total_reward)
        epsilon_values.append(epsilon)
        steps_per_episode.append(steps)  # Store steps per episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if ep % 50 == 0:
            print(f"Episode {ep}, Epsilon: {epsilon:.3f}, Reward: {total_reward}, Steps: {steps}")


def reset_environment():
    global drone_start, walls, obstacle_objects
    goal_pos[:] = [ROWS - 2, COLS - 2]  # Optional: randomize goal too if desired

    # Randomize drone start safely
    while True:
        drone_start = [random.randint(0, ROWS - 1), random.randint(0, COLS - 1)]
        if drone_start != goal_pos:
            break

    # Rebuild obstacles
    walls = []
    obstacle_objects = []
    for _ in range(60):
        while True:
            pos = [random.randint(0, ROWS - 1), random.randint(0, COLS - 1)]
            if pos != drone_start and pos != goal_pos and pos not in walls:
                walls.append(pos)
                img = random.choice(obstacle_imgs)
                obstacle_objects.append((pos, img))
                break

# RL Simulation
def simulate():
    drone_pos = drone_start[:]
    path = [drone_pos[:]]
    steps = 0
    loop_counter = {}

    draw_game(drone_pos, path, steps)
    
    # ✅ Wait 1 second before movement starts
    pygame.time.wait(1000)

    for _ in range(200):
        clock.tick(3)

        # ✅ Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        row, col = drone_pos
        best_action = np.argmax(q_table[row, col])  # Choose best action (lowest step policy)
        dx, dy = actions[best_action]
        new_row, new_col = row + dy, col + dx

        # ✅ If move is valid
        if 0 <= new_row < ROWS and 0 <= new_col < COLS and [new_row, new_col] not in walls:
            drone_pos = [new_row, new_col]
            path.append(drone_pos[:])
            steps += 1

            # ✅ Fix: define pos_tuple before using it
            pos_tuple = tuple(drone_pos)
            loop_counter[pos_tuple] = loop_counter.get(pos_tuple, 0) + 1

            # ✅ Loop detection
            if loop_counter[pos_tuple] > 3:
                print("Stuck in a loop! Ending simulation.")
                pygame.time.wait(1000)
                pygame.quit()
                sys.exit()

        draw_game(drone_pos, path, steps)

        if drone_pos == goal_pos:
            print(f"Goal Reached in {steps} steps!")
            global last_path
            last_path = path[:]
            pygame.time.wait(800)
            draw_game(drone_pos, path, steps)

            pygame.display.flip()

            pygame.time.wait(4000)
            pygame.quit()
            # sys.exit()
            plot_training_stats()


def plot_training_stats():
    plt.figure(figsize=(14, 8))

    # 1. Rewards per Episode
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Reward per Episode", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Episodes")
    plt.grid(True)

    # 2. Epsilon Decay
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values, label="Epsilon", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Time")
    plt.grid(True)

    # 3. Steps per Episode (efficiency indicator)
    plt.subplot(2, 2, 3)
    plt.plot(steps_per_episode, label="Steps per Episode", color='orange')
    plt.xlabel("Episodes")
    plt.ylabel("Steps Taken")
    plt.title("Steps Taken per Episode")
    plt.grid(True)

    # 4. Moving Average of Rewards
    plt.subplot(2, 2, 4)
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f"Moving Avg (window={window})", color='purple')
        plt.title("Moving Average of Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Avg Reward")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Not enough data for moving average', ha='center')

    plt.tight_layout()
    plt.show()
    max_q = np.max(q_table, axis=2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(max_q, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Max Q-value Per Cell (After Training)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()

def plot_q_table_heatmap():
    # Separate heatmap for Q-table
    max_q = np.max(q_table, axis=2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(max_q, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Max Q-value Per Cell (After Training)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()



# Main
if __name__ == '__main__':
    show_home_screen() 
    train_q_learning()
    simulate()
    plot_training_stats()
    plot_q_table_heatmap()
    
    
