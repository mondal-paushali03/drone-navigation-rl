# 2D Drone Navigation using Q-Learning

A Python-based 2D drone navigation simulation where a drone learns to reach a goal while avoiding obstacles using **Reinforcement Learning (Q-Learning)**. The project leverages **Pygame** for visualization and **NumPy/Seaborn/Matplotlib** for analytics and training insights.
<img width="800" height="800" alt="Screenshot 2025-08-22 144243" src="https://github.com/user-attachments/assets/66736b10-445f-46e6-b7d1-b4227991f5e0" />

---

## Features

- **Interactive Home Screen:** Welcome screen with a start prompt.
- **Drone Navigation:** Drone moves on a 2D grid with obstacles and a goal.
- **Obstacle Avoidance:** Randomly placed obstacles with different images.
- **Reinforcement Learning:** 
  - Q-Learning algorithm for training optimal paths.
  - Epsilon-greedy policy for exploration-exploitation balance.
  - Reward shaping based on distance to goal and revisiting locations.
- **Simulation:** Visual step-by-step drone movement with loop detection.
- **Analytics:**
  - Episode rewards over training.
  - Epsilon decay visualization.
  - Steps per episode for efficiency.
  - Moving average of rewards.
  - Q-table heatmap for maximum Q-values per cell.

---


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mondal-paushali03/drone-navigation-rl.git
   cd drone-navigation-rl
2. **Install dependencies:**
    ```bash
    pip install pygame numpy matplotlib seaborn
3. **Ensure `Assets` folder exists:**

- `background.jpg` → Main simulation background  
- `bg1.png` → Home screen background  
- `drone.png` → Drone image  
- `goal.png` → Goal image  
- `obstacle1.png`, `obstacle2.png`, `obstacle3.png` → Obstacle images

---
## Reinforcement Learning Details

- **State Space:** Each grid cell on the 2D map.
- **Action Space:** Move Up, Down, Left, Right.

### Reward Function
- `+100` for reaching goal  
- `-100` for hitting walls  
- `-1` per step  
- `-10` for revisiting the same cell  
- `+1/-1` for moving closer/farther from the goal (distance-based reward shaping)

### Hyperparameters
- **Learning Rate (α):** 0.1  
- **Discount Factor (γ):** 0.9  
- **Exploration Rate (ε):** Starts at 1.0, decays to 0.1  
- **Episodes:** 3000
<img width="1100" height="800" alt="Screenshot 2025-08-22 144313" src="https://github.com/user-attachments/assets/1b751ee8-b6fa-406d-b99d-c09e64fe9a90" />

## Contact
For questions or analysis, contact Paushali Mondal at mondal.paushali384@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
