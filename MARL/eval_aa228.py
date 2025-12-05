# Evaluation script for AA228 MAPPO model
import cv2
import gym 
import torch
import numpy as np
import os
from MAPPO import MAPPO
import highway_env

# MODEL_NAME = "Dec_04_01_52_09_8agents"
# MODEL_NAME = "Dec_04_07_42_48_4agents"
MODEL_NAME = "Dec_04_08_58_45_1agent"
MODEL_PATH = "./results_aa228/" + MODEL_NAME + "/models/checkpoint-2000.pt"  # Update as needed
CONFIG_PATH = "./configs/configs_ppo.ini"  # Update as needed
ENV_ID = "aa228-multi-agent-v0"
NUM_EPISODES = 30

def load_config(config_path):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    config = load_config(CONFIG_PATH)
    env = gym.make(ENV_ID)
    
    # --- Environment Configuration ---
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    state_dim = env.n_s
    action_dim = env.n_a

    mappo = MAPPO(env=env, memory_capacity=1000,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=32, entropy_reg=0.01,
                  roll_out_n_steps=1,
                  actor_hidden_size=128, critic_hidden_size=128,
                  actor_lr=1e-4, critic_lr=1e-4, reward_scale=20,
                  target_update_steps=5, target_tau=1.0,
                  reward_gamma=0.99, reward_type="global_R",
                  max_grad_norm=0.5, test_seeds="0",
                  episodes_before_train=1, traffic_density=env.config['traffic_density'])

    # Load trained model
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        mappo.actor.load_state_dict(checkpoint['model_state_dict'])
        mappo.actor.eval()
    else:
        print(f"Model not found at {MODEL_PATH}")
        return

    # --- UPDATED METRICS STORAGE ---
    all_agent_speeds = []     # Stores Average Speed (Distance/Time) for ALL non-collided agents
    collided_episodes = 0     # Counter for episodes with at least one collision
    all_episode_metrics = []  # Stores list of speeds/0 for each agent per episode
    # -------------------------------

    for ep in range(NUM_EPISODES):
        state, action_mask = env.reset()
        done = False
        n_agents = len(env.controlled_vehicles)
        print(f"\n--- Episode {ep+1} / {NUM_EPISODES} ---")
        
        agent_time = [0 for _ in range(n_agents)]
        agent_distance = [0.0 for _ in range(n_agents)]
        agent_collided = [False for _ in range(n_agents)]
        agent_goal = [False for _ in range(n_agents)]
        prev_positions = [np.array(v.position) for v in env.controlled_vehicles]
        video_file = f"eval_video/{MODEL_NAME}/eval_episode_{ep+1}.mp4"
        os.makedirs(os.path.dirname(video_file), exist_ok=True)
        frame = env.render(mode="rgb_array")
        height, width, channels = frame.shape
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
        
        episode_collision = False # Flag for this specific episode

        while not done:
            actions = mappo.action(state, n_agents)
            next_state, reward, done, info = env.step(tuple(actions))
            for i, v in enumerate(env.controlled_vehicles):
                if agent_collided[i] or agent_goal[i]:
                    continue
                agent_time[i] += 1
                curr_pos = np.array(v.position)
                agent_distance[i] += np.linalg.norm(curr_pos - prev_positions[i])
                prev_positions[i] = curr_pos
                if v.crashed:
                    agent_collided[i] = True
                    episode_collision = True # Mark the episode as having a collision
                # Manual goal detection: if distance > 200, mark as goal
                if agent_distance[i] > 200:
                    agent_goal[i] = True
                # Original goal detection (keep for reference)
                if hasattr(v, 'route') and v.route:
                    if v.route[-1][1] == 'r_out':
                        lane = env.road.network.get_lane(v.lane_index)
                        if v.lane_index[1] == 'r_out' and lane.after_end(v.position):
                            agent_goal[i] = True
            # Render and write frame to video
            frame = env.render(mode="rgb_array")
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            state = next_state

        # --- EPISODE END: METRIC CALCULATION ---
        if episode_collision:
            collided_episodes += 1
        
        current_episode_metrics = []
        
        # Print and store per-agent results for this episode
        for i in range(n_agents):
            print(f"  Agent {i}: time={agent_time[i]}, distance={agent_distance[i]:.2f}, collided={agent_collided[i]}, goal={agent_goal[i]}")
            
            if agent_collided[i]:
                # Collision -> 0 speed
                metric_val = 0.0
                print(f"    -> Metric (Avg Speed): 0.0 m/s (collided)")
            elif agent_time[i] > 0:
                # Success/Runout -> Calculate Average Speed (Distance / Time)
                avg_speed = agent_distance[i] / agent_time[i]
                all_agent_speeds.append(avg_speed) # STORED HERE
                metric_val = avg_speed
                if agent_goal[i]:
                    print(f"    -> Metric (Avg Speed): {avg_speed:.2f} m/s (reached goal)")
                else:
                    print(f"    -> Metric (Avg Speed): {avg_speed:.2f} m/s (non-collided, episode ended)")
            else:
                 # Should not happen in a typical run, but for safety
                metric_val = 0.0

            current_episode_metrics.append(metric_val)

        all_episode_metrics.append(current_episode_metrics)
        video_writer.release()
        print(f"Episode {ep+1} video saved to {video_file}")

    # --- FINAL ANALYSIS ---
    
    # 1. Safety Metric: Collision Rate
    collision_rate = (collided_episodes / NUM_EPISODES) * 100
    
    # 2. Efficiency Metric: Average Speed (Calculated in two ways)
    
    # a) Average Speed over all non-collided agent runs
    if all_agent_speeds:
        avg_speed_non_collided = np.mean(all_agent_speeds)
        # --- NEW: MAX and MIN SPEED ---
        max_speed_non_collided = np.max(all_agent_speeds)
        min_speed_non_collided = np.min(all_agent_speeds)
        # -----------------------------
    else:
        avg_speed_non_collided = 0.0
        max_speed_non_collided = 0.0
        min_speed_non_collided = 0.0
    
    # b) Average Speed over agents in fully collision-free episodes
    non_collision_episode_speeds = []
    non_collided_episode_list = []
    
    for i, ep_metrics in enumerate(all_episode_metrics):
        # Check if ALL agents in the episode have a speed > 0 (i.e., no collisions)
        if all(speed > 0.0 for speed in ep_metrics):
            non_collision_episode_speeds.extend(ep_metrics)
            non_collided_episode_list.append(i + 1)
            
    avg_speed_clean_episodes = np.mean(non_collision_episode_speeds) if non_collision_episode_speeds else 0.0

    print("\n" + "="*50)
    print("🚦 FINAL EVALUATION RESULTS (Average Speed & Safety)")
    print("="*50)
    print(f"Total Episodes Evaluated: {NUM_EPISODES}")
    
    # --- SAFETY METRIC (Minimize) ---
    print("\n### 🛡️ Safety Metric")
    print(f"Episodes with Collision: **{collided_episodes}** / {NUM_EPISODES}")
    print(f"Collision Rate: **{collision_rate:.2f}%**")
    print(f"**Collision-Free Episodes:** {non_collided_episode_list}")
    
    # --- EFFICIENCY METRIC (Maximize) ---
    print("\n### 🚀 Efficiency Metric (Average Speed)")
    print(f"Total non-collided agent runs: {len(all_agent_speeds)}")
    print(f"Average Speed (All non-collided agents): **{avg_speed_non_collided:.2f} m/s**")
    print(f"Max Average Speed (All non-collided agents): **{max_speed_non_collided:.2f} m/s**")
    print(f"Min Average Speed (All non-collided agents): **{min_speed_non_collided:.2f} m/s**")
    print(f"Average Speed (Agents in fully collision-free episodes): **{avg_speed_clean_episodes:.2f} m/s**")
    
if __name__ == "__main__":
    main()