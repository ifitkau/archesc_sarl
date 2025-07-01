"""
Multi-Room Escape Environment for Reinforcement Learning

This module implements a custom reinforcement learning environment where an agent must navigate
through a multi-room maze, create doors between rooms, and reach a terminal goal location.
The environment is built on top of the MiniWorld framework and is designed for training
navigation and exploration behaviors.

Key Features:
- 4-room layout with hallway connection system
- Dynamic door creation when agent touches walls between rooms
- Comprehensive reward system for navigation, exploration, and goal achievement
- Multiple observation types including lidar measurements and room categorization
- Extensive logging and tracking capabilities for training analysis

Date: July 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from miniworld.entity import Agent, Box, COLORS
from miniworld.miniworld import MiniWorldEnv, Room
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from wandb.integration.sb3 import WandbCallback
import multiprocessing
import wandb
import random
from tabulate import tabulate
import itertools
import gc
import traceback
import time
import os
import shutil


class ImprovedTrackerCallback(BaseCallback):
    """
    Advanced callback that tracks door creation events and terminal achievements across episodes.
    
    This callback logs detailed information about:
    - Door creation positions and timing
    - Terminal state achievements
    - Episode outcomes and success rates
    - Room exploration patterns
    
    Creates separate CSV files for different types of events, making it easy to analyze
    agent behavior and learning progress over time.
    """
    def __init__(self, log_dir="./logs", verbose=0):
        super(ImprovedTrackerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Track data for each environment separately (for vectorized environments)
        self.env_data = {}
        self.log_files = {}
        self.door_csv_files = {}
        self.terminal_csv_files = {}
        self.episode_outcomes_files = {}
        self.timestamp = int(time.time())
    
    def _on_training_start(self):
        """Initialize CSV files with appropriate headers for data logging"""
        # Initialize CSV files with headers
        for env_id in range(self.model.get_env().num_envs):
            # Create door creation CSV file
            door_csv_path = f"{self.log_dir}/door_creations_env{env_id}_{self.timestamp}.csv"
            self.door_csv_files[env_id] = door_csv_path
            
            # Create terminal events CSV file
            terminal_csv_path = f"{self.log_dir}/terminal_events_env{env_id}_{self.timestamp}.csv"
            self.terminal_csv_files[env_id] = terminal_csv_path
            
            # Create episode outcomes file
            episode_outcomes_path = f"{self.log_dir}/episode_outcomes_env{env_id}_{self.timestamp}.csv"
            self.episode_outcomes_files[env_id] = episode_outcomes_path
            
            with open(door_csv_path, "w") as f:
                # Add terminal_reached to door CSV header
                f.write("episode,step,x,y,z,room,from_room,to_room,is_door,terminal_reached\n")
                
            with open(terminal_csv_path, "w") as f:
                f.write("episode,step,x,y,z,room,reward,total_steps\n")
                
            with open(episode_outcomes_path, "w") as f:
                f.write("episode,terminal_reached,steps,rooms_visited\n")
    
    def _on_step(self) -> bool:
        """
        Process each environment step and log relevant events.
        
        This method is called at every environment step and handles:
        - Door creation event logging
        - Terminal state detection and logging
        - Episode boundary detection and data aggregation
        - Room exploration tracking
        """
        # We need to process all environments in the vectorized env
        for i, info in enumerate(self.locals['infos']):
            env_id = i  # Environment ID in vectorized env
            
            # Initialize environment data if not exists
            if env_id not in self.env_data:
                self.env_data[env_id] = {
                    "episode_count": 0,
                    "current_episode": 0,
                    "terminal_reached": False,
                    "rooms_visited": set(),
                    "current_episode_step": 0
                }
                
                # Initialize the log file with headers if not exists
                self.log_files[env_id] = f"{self.log_dir}/tracking_env{env_id}_{self.timestamp}.txt"
                if not os.path.exists(self.log_files[env_id]):
                    with open(self.log_files[env_id], "w") as f:
                        f.write(f"=== DOOR AND REWARD TRACKING LOG - ENVIRONMENT {env_id} ===\n")
                        f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n\n")
            
            # Get the environment instance
            env = self.model.get_env().envs[env_id].unwrapped
            
            # Update step counter for the current episode
            self.env_data[env_id]["current_episode_step"] += 1
            
            # Check if terminal was reached this episode - check multiple sources for reliability
            if info.get('reached_terminal', False) or (hasattr(env, 'reached_terminal_area') and env.reached_terminal_area):
                self.env_data[env_id]["terminal_reached"] = True
                
            # Track rooms visited during this episode
            current_room = env._get_current_room()
            self.env_data[env_id]["rooms_visited"].add(current_room)
            
            # 1. Track actual door creation events (between rooms, not terminal connections)
            if ('created_door' in info and info['created_door'] and 
                'door_connection_from' in info and 'door_connection_to' in info and
                info['door_connection_to'] != 'Terminal'):
                
                # Get door connection information
                from_room = f"room{info['door_connection_from']}"
                to_room = f"room{info['door_connection_to']}"
                
                current_room = env._get_current_room()
                pos = [round(p, 2) for p in env.agent.pos]
                
                # We'll use a "?" for terminal_reached since we don't know yet
                # We'll update this at the end of the episode
                with open(self.door_csv_files[env_id], "a") as f:
                    f.write(f"{self.env_data[env_id]['current_episode']},{env.step_count},{pos[0]},{pos[1]},{pos[2]},{current_room},{from_room},{to_room},True,?\n")
                
                # Also log to text file for human readability
                with open(self.log_files[env_id], "a") as f:
                    f.write(f"Episode {self.env_data[env_id]['current_episode']} - Step {env.step_count}: Door created at position {pos} in {current_room} (from {from_room} to {to_room})\n")
            
            # 2. Separately track terminal state events
            if info.get('reached_terminal', False) or (hasattr(env, 'reached_terminal_area') and env.reached_terminal_area):
                current_room = env._get_current_room()
                pos = [round(p, 2) for p in env.agent.pos]
                reward = info.get('reward_terminal', 0)
                
                # Write to terminal events CSV file - include the total steps
                with open(self.terminal_csv_files[env_id], "a") as f:
                    f.write(f"{self.env_data[env_id]['current_episode']},{env.step_count},{pos[0]},{pos[1]},{pos[2]},{current_room},{reward},{self.env_data[env_id]['current_episode_step']}\n")
                
                # Also log to text file for human readability
                with open(self.log_files[env_id], "a") as f:
                    f.write(f"Episode {self.env_data[env_id]['current_episode']} - Step {env.step_count}: TERMINAL REACHED at position {pos} in {current_room} (reward: {reward}, total steps: {self.env_data[env_id]['current_episode_step']})\n")
            
            # Check if this is a new episode - if so, write the previous episode's data and update door creation records
            if 'episode' in info:
                # Write episode outcome data if it's not the first episode
                if self.env_data[env_id]["current_episode"] > 0:
                    # Record episode outcome
                    terminal_reached = 1 if self.env_data[env_id]["terminal_reached"] else 0
                    rooms_visited = len(self.env_data[env_id]["rooms_visited"])
                    total_steps = self.env_data[env_id]["current_episode_step"]
                    
                    with open(self.episode_outcomes_files[env_id], "a") as f:
                        f.write(f"{self.env_data[env_id]['current_episode']},{terminal_reached},{total_steps},{rooms_visited}\n")
                    
                    # Now update all door creation records for this episode with the terminal status
                    self._update_door_creation_terminal_status(env_id, self.env_data[env_id]["current_episode"], terminal_reached)
                
                # Reset for new episode
                self.env_data[env_id]["episode_count"] += 1
                self.env_data[env_id]["current_episode"] = self.env_data[env_id]["episode_count"]
                self.env_data[env_id]["terminal_reached"] = False
                self.env_data[env_id]["rooms_visited"] = set()
                self.env_data[env_id]["current_episode_step"] = 0
            
            # Check for episode termination to log episode boundary
            if info.get('reached_terminal', False) or info.get('max_steps_reached', False):
                with open(self.log_files[env_id], "a") as f:
                    terminal_status = "TERMINAL REACHED" if self.env_data[env_id]["terminal_reached"] else "TERMINAL NOT REACHED"
                    f.write(f"\nEnd of Episode {self.env_data[env_id]['current_episode']} - {terminal_status} - Total Steps: {self.env_data[env_id]['current_episode_step']}\n\n")
        
        return True
    
    def _update_door_creation_terminal_status(self, env_id, episode, terminal_reached):
        """
        Update door creation records for a completed episode with the terminal status.
        
        This method goes back and fills in the terminal_reached status for all door
        creation events that occurred during the episode, since we only know the
        episode outcome at the end.
        """
        # Read all lines from the door creation CSV
        door_csv_path = self.door_csv_files[env_id]
        with open(door_csv_path, 'r') as f:
            lines = f.readlines()
        
        # Process each line, updating the terminal_reached field for the given episode
        updated_lines = []
        for i, line in enumerate(lines):
            if i == 0:  # Header line
                updated_lines.append(line)
                continue
                
            parts = line.strip().split(',')
            if len(parts) >= 10 and int(parts[0]) == episode and parts[9] == '?':
                # Update the terminal_reached field (last column)
                parts[9] = str(terminal_reached)
                updated_lines.append(','.join(parts) + '\n')
            else:
                updated_lines.append(line)
        
        # Write the updated lines back to the file
        with open(door_csv_path, 'w') as f:
            f.writelines(updated_lines)

class TrackerCallback(BaseCallback):
    """
    Comprehensive callback that tracks detailed reward components and agent behavior.
    
    This callback provides in-depth analysis of:
    - Reward component breakdown by episode
    - Agent position tracking over time
    - Door creation events with full context
    - Significant reward events that impact learning
    
    Creates human-readable text files with detailed episode summaries.
    """
    def __init__(self, log_dir="./logs", verbose=0):
        super(TrackerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Track data for each environment separately
        self.env_data = {}
        self.log_files = {}
        self.timestamp = int(time.time())
    
    def _on_step(self) -> bool:
        """Process each environment step and collect detailed tracking data"""
        # We need to process all environments in the vectorized env
        for i, info in enumerate(self.locals['infos']):
            env_id = i  # Environment ID in vectorized env
            
            # Initialize environment data if not exists
            if env_id not in self.env_data:
                self.env_data[env_id] = {
                    "episode_count": 0,
                    "current_episode_data": {
                        "doors": [],
                        "rewards": [],
                        "positions": []
                    }
                }
                
                # Create a separate log file for each environment
                self.log_files[env_id] = f"{self.log_dir}/tracking_env{env_id}_{self.timestamp}.txt"
                
                # Initialize the log file with headers
                with open(self.log_files[env_id], "w") as f:
                    f.write(f"=== DOOR AND REWARD TRACKING LOG - ENVIRONMENT {env_id} ===\n")
                    f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n\n")
            
            # Check if this is a new episode
            if 'episode' in info:
                # Save previous episode data if it exists
                if (self.env_data[env_id]["current_episode_data"]["rewards"] or 
                    self.env_data[env_id]["current_episode_data"]["doors"]):
                    self._save_episode_data(env_id)
                    self.env_data[env_id]["current_episode_data"] = {
                        "doors": [],
                        "rewards": [],
                        "positions": []
                    }
                self.env_data[env_id]["episode_count"] += 1
            
            # Track door creation events with full context
            if 'created_door' in info and info['created_door']:
                # Get the environment instance
                env = self.model.get_env().envs[env_id].unwrapped
                
                # Get door connection information - check in info first, then environment
                if 'door_connection_from' in info and 'door_connection_to' in info:
                    connection_info = f"from room{info['door_connection_from']} to room{info['door_connection_to']}"
                else:
                    connection_info = self._get_latest_connection_info(env)
                
                # Check if this is a terminal door
                is_terminal = info.get('terminal_door_creation', False) or info.get('reached_terminal', False)
                if is_terminal:
                    connection_info = f"from {env._get_current_room()} to Terminal State"
                
                door_data = {
                    "step": env.step_count,
                    "position": [round(pos, 2) for pos in env.agent.pos],
                    "room": env._get_current_room(),
                    "connection": connection_info,
                    "is_terminal": is_terminal
                }
                self.env_data[env_id]["current_episode_data"]["doors"].append(door_data)
            
            # Get the environment instance to access reward scales
            env = self.model.get_env().envs[env_id].unwrapped
            
            # Track reward components in each step (using scaled values)
            raw_reward_components = {
                key: info[key] for key in [
                    'reward_orientation',
                    'reward_distance', 'punishment_distance', 
                    'penalty_stagnation', 'punishment_time',
                    'reward_hallway', 'reward_connection', 
                    'reward_terminal', 'punishment_terminal',
                    'punishment_room', 'episode_rewards'
                ] if key in info
            }
            
            # Apply reward scales to get the actual contribution to total reward
            scaled_reward_components = {}
            if raw_reward_components:
                # Apply reward scales to convert raw rewards to actual contributions
                if 'reward_orientation' in raw_reward_components:
                    scaled_reward_components['reward_orientation'] = raw_reward_components['reward_orientation'] * env.reward_scales['reward_orientation_scale']

                if 'reward_distance' in raw_reward_components:
                    scaled_reward_components['reward_distance'] = raw_reward_components['reward_distance'] * env.reward_scales['reward_distance_scale']
                
                if 'punishment_distance' in raw_reward_components:
                    scaled_reward_components['punishment_distance'] = raw_reward_components['punishment_distance'] * env.reward_scales['punishment_distance_scale']
                
                if 'penalty_stagnation' in raw_reward_components:
                    scaled_reward_components['penalty_stagnation'] = raw_reward_components['penalty_stagnation'] * env.reward_scales['penalty_stagnation_scale']
                
                if 'punishment_time' in raw_reward_components:
                    scaled_reward_components['punishment_time'] = raw_reward_components['punishment_time'] * env.reward_scales['punishment_time_scale']
                
                if 'reward_hallway' in raw_reward_components:
                    scaled_reward_components['reward_hallway'] = raw_reward_components['reward_hallway'] * env.reward_scales['reward_hallway_scale']
                
                if 'reward_connection' in raw_reward_components:
                    scaled_reward_components['reward_connection'] = raw_reward_components['reward_connection'] * env.reward_scales['reward_connection_scale']
                
                if 'reward_terminal' in raw_reward_components:
                    scaled_reward_components['reward_terminal'] = raw_reward_components['reward_terminal'] * env.reward_scales['reward_terminal_scale']
                
                if 'punishment_terminal' in raw_reward_components:
                    scaled_reward_components['punishment_terminal'] = raw_reward_components['punishment_terminal'] * env.reward_scales['punishment_terminal_scale']
                
                if 'punishment_room' in raw_reward_components:
                    scaled_reward_components['punishment_room'] = raw_reward_components['punishment_room'] * env.reward_scales['punishment_room_scale']
                
                # Include total episode rewards
                if 'episode_rewards' in raw_reward_components:
                    scaled_reward_components['episode_rewards'] = raw_reward_components['episode_rewards']
                
                reward_data = {
                    "step": info.get('step_count', env.step_count),
                    "components": scaled_reward_components,
                    "room": info.get('current_room', env._get_current_room())
                }
                self.env_data[env_id]["current_episode_data"]["rewards"].append(reward_data)
            
            # Track position every 10 steps for trajectory analysis
            if hasattr(env, 'step_count') and env.step_count % 10 == 0:
                pos_data = {
                    "step": env.step_count,
                    "position": [round(pos, 2) for pos in env.agent.pos],
                    "room": env._get_current_room()
                }
                self.env_data[env_id]["current_episode_data"]["positions"].append(pos_data)
            
            # Check for episode termination
            if info.get('reached_terminal', False) or info.get('max_steps_reached', False):
                # Add a terminal door creation if terminal state was reached and no doors recorded
                if info.get('reached_terminal', False) and not any(door.get('is_terminal', False) 
                                                               for door in self.env_data[env_id]["current_episode_data"]["doors"]):
                    door_data = {
                        "step": env.step_count,
                        "position": [round(pos, 2) for pos in env.agent.pos],
                        "room": env._get_current_room(),
                        "connection": f"from {env._get_current_room()} to Terminal State",
                        "is_terminal": True
                    }
                    self.env_data[env_id]["current_episode_data"]["doors"].append(door_data)
                
                self._save_episode_data(env_id)
                self.env_data[env_id]["current_episode_data"] = {
                    "doors": [],
                    "rewards": [],
                    "positions": []
                }
        
        return True
    
    def _get_latest_connection_info(self, env):
        """Extract connection information from the environment state"""
        # Use the last_created_connection if available
        if hasattr(env, 'last_created_connection'):
            from_room = env.last_created_connection["from"]
            to_room = env.last_created_connection["to"]
            return f"from room{from_room} to room{to_room}"
        
        # Otherwise, search through connections
        for (x_range, z_range), data in env.connections.items():
            if data['created']:
                connection_str = data['connection']
                rooms = connection_str.split('-')
                return f"from room{rooms[0]} to room{rooms[1]}"
        
        return "unknown connection"
    
    def _save_episode_data(self, env_id):
        """
        Save comprehensive episode data to the log file.
        
        Creates a detailed summary including:
        - Door creation events and timing
        - Agent position trajectory
        - Reward component breakdown
        - Significant events that affected learning
        """
        # Get the log file for this environment
        log_file = self.log_files[env_id]
        
        with open(log_file, "a") as f:
            f.write(f"\n\n=== EPISODE {self.env_data[env_id]['episode_count']} ===\n")
            
            # Write door creation events
            f.write("\n--- DOOR CREATIONS ---\n")
            if not self.env_data[env_id]["current_episode_data"]["doors"]:
                f.write("No doors created in this episode.\n")
            else:
                for door in self.env_data[env_id]["current_episode_data"]["doors"]:
                    # Check if this is a terminal door
                    if door.get('is_terminal', False):
                        f.write(f"Step {door['step']}: TERMINAL CONNECTION created at position {door['position']} in {door['room']} ({door['connection']})\n")
                    else:
                        f.write(f"Step {door['step']}: Door created at position {door['position']} in {door['room']} ({door['connection']})\n")
            
            # Write position tracking - sort by step number
            f.write("\n--- AGENT POSITIONS (every 10 steps) ---\n")
            if not self.env_data[env_id]["current_episode_data"]["positions"]:
                f.write("No position data recorded.\n")
            else:
                # Sort positions by step number
                sorted_positions = sorted(self.env_data[env_id]["current_episode_data"]["positions"], key=lambda x: x["step"])
                for pos in sorted_positions:
                    f.write(f"Step {pos['step']}: Position {pos['position']} in {pos['room']}\n")
            
            # Write reward summaries
            f.write("\n--- REWARD SUMMARY (SCALED VALUES) ---\n")
            if not self.env_data[env_id]["current_episode_data"]["rewards"]:
                f.write("No rewards recorded in this episode.\n")
            else:
                # Calculate totals for each reward component
                totals = {}
                for reward_entry in self.env_data[env_id]["current_episode_data"]["rewards"]:
                    for component, value in reward_entry["components"].items():
                        if component not in totals:
                            totals[component] = 0
                        totals[component] += value
                
                # Write the totals
                f.write("Total reward components for episode (scaled values):\n")
                for component, value in totals.items():
                    # Only write components with non-zero values
                    if abs(value) > 0.001:  # Skip components with effectively zero value
                        f.write(f"  {component}: {value:.2f}\n")
                
                # Write significant rewards (absolute value > 10)
                f.write("\nSignificant reward events (|value| > 10):\n")
                significant_events = False
                
                # Sort rewards by step number
                sorted_rewards = sorted(self.env_data[env_id]["current_episode_data"]["rewards"], key=lambda x: x["step"])
                
                for reward_entry in sorted_rewards:
                    for component, value in reward_entry["components"].items():
                        if abs(value) > 10:
                            f.write(f"  Step {reward_entry['step']} in {reward_entry['room']}: {component} = {value:.2f}\n")
                            significant_events = True
                
                if not significant_events:
                    f.write("  No significant reward events in this episode.\n")

class RewardSumCallback(BaseCallback):
    """
    Basic callback for tracking cumulative reward statistics and logging to Weights & Biases.
    
    Tracks:
    - Average reward per step and per episode
    - Terminal state achievement percentage
    - Episode completion statistics
    """
    def __init__(self, verbose=0):
        super(RewardSumCallback, self).__init__(verbose)
        self.rewards_sum = 0
        self.steps_sum = 0  
        self.episode_count = 0
        self.terminal_reached_count = 0  # Track how many episodes reached terminal
        
    def _on_step(self) -> bool:
        """Update reward statistics and log to Weights & Biases"""
        self.steps_sum += 1
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                if 'episode_rewards' in info:
                    current_reward = info['episode_rewards']
                    self.rewards_sum += current_reward
                    self.episode_count += 1
                    
                    # Track terminal reached
                    if info.get('terminal_reached_this_episode', False):
                        self.terminal_reached_count += 1

        # Calculate averages
        average_reward_per_step = self.rewards_sum / self.steps_sum if self.steps_sum > 0 else 0
        average_reward_per_episode = self.rewards_sum / self.episode_count if self.episode_count > 0 else 0
        terminal_reached_percentage = (self.terminal_reached_count / self.episode_count * 100) if self.episode_count > 0 else 0

        # Log the average reward to WandB
        wandb.log({
            "average_reward_per_step": average_reward_per_step,
            "average_reward_per_episode": average_reward_per_episode,
            "terminal_reached_percentage": terminal_reached_percentage,
            "steps_sum": self.steps_sum,
            "episodes_completed": self.episode_count,
            "terminal_reached_count": self.terminal_reached_count
        })

        return True
    
class EnhancedRewardCallback(BaseCallback):
    """
    Enhanced callback that provides detailed episode analysis and iteration-based reporting.
    
    This callback extends basic reward tracking with:
    - Per-iteration episode statistics
    - Terminal achievement tracking
    - Detailed logging to files and Weights & Biases
    - Training progress analysis
    """
    def __init__(self, log_dir, timesteps_per_iteration, verbose=0):
        super(EnhancedRewardCallback, self).__init__(verbose)
        self.rewards_sum = 0
        self.steps_sum = 0  
        self.episode_count = 0
        self.terminal_reached_count = 0
        
        # For file logging
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time())
        self.log_file = f"{log_dir}/episodes_per_iteration_{timestamp}.txt"
        self.timesteps_per_iteration = timesteps_per_iteration
        
        # Track episodes per iteration
        self.iteration_data = {}  # {iteration: [episode_count, terminal_count, reward_sum]}
        self.episode_buffer = []  # Buffer to collect all episodes before assigning to iterations
        
        # Write header to file
        with open(self.log_file, "w") as f:
            f.write("Iteration,Episodes,TerminalReached,AverageReward\n")
        
    def _on_step(self) -> bool:
        """Track episode completions and update statistics"""
        # Basic step and reward tracking for wandb
        self.steps_sum += 1
        
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                # Check if an episode has completed
                if 'episode' in info:
                    # Get reward from stable-baselines format
                    current_reward = info['episode']['r']
                    self.rewards_sum += current_reward
                    self.episode_count += 1
                    
                    # Track terminal reached
                    terminal_reached = info.get('reached_terminal', False) or info.get('terminal_reached_this_episode', False)
                    if terminal_reached:
                        self.terminal_reached_count += 1
                    
                    # Store episode in buffer for later iteration assignment
                    self.episode_buffer.append({
                        'timestep': self.num_timesteps,
                        'terminal_reached': terminal_reached,
                        'reward': current_reward
                    })

        # Calculate averages for wandb logging
        average_reward_per_step = self.rewards_sum / self.steps_sum if self.steps_sum > 0 else 0
        average_reward_per_episode = self.rewards_sum / self.episode_count if self.episode_count > 0 else 0
        terminal_reached_percentage = (self.terminal_reached_count / self.episode_count * 100) if self.episode_count > 0 else 0

        # Log the metrics to WandB with correct step counting
        wandb.log({
            "average_reward_per_step": average_reward_per_step,
            "average_reward_per_episode": average_reward_per_episode,
            "terminal_reached_percentage": terminal_reached_percentage,
            "steps_sum": self.steps_sum,
            "episodes_completed": self.episode_count,
            "terminal_reached_count": self.terminal_reached_count
        }, step=self.num_timesteps)

        return True
        
    def _assign_episodes_to_iterations(self):
        """
        Assign episodes to training iterations based on timestep thresholds.
        
        This method organizes episode data by training iteration, allowing for
        analysis of learning progress over the course of training.
        """
        # Sort episodes by timestep to ensure proper ordering
        self.episode_buffer.sort(key=lambda x: x['timestep'])
        
        # Clear previous iteration data
        self.iteration_data = {}
        
        # Process each episode
        for episode in self.episode_buffer:
            iteration = episode['timestep'] // self.timesteps_per_iteration
            
            # Initialize iteration data if needed
            if iteration not in self.iteration_data:
                self.iteration_data[iteration] = [0, 0, 0.0]  # [episode_count, terminal_count, reward_sum]
            
            # Increment counters
            self.iteration_data[iteration][0] += 1  # episode count
            if episode['terminal_reached']:
                self.iteration_data[iteration][1] += 1  # terminal count
            self.iteration_data[iteration][2] += episode['reward']  # reward sum
    
    def on_training_end(self):
        """Generate final training report with iteration-based statistics"""
        # Process all episodes into iterations
        self._assign_episodes_to_iterations()
        
        # Write per-iteration statistics
        for iteration in sorted(self.iteration_data.keys()):
            episodes, terminals, reward_sum = self.iteration_data[iteration]
            avg_reward = reward_sum / episodes if episodes > 0 else 0.0
            
            with open(self.log_file, "a") as f:
                f.write(f"{iteration},{episodes},{terminals},{avg_reward:.4f}\n")
        
        # Write summary statistics
        with open(self.log_file, "a") as f:
            f.write("\n--- SUMMARY ---\n")
            
            # Calculate totals
            total_episodes = sum(data[0] for data in self.iteration_data.values())
            total_terminals = sum(data[1] for data in self.iteration_data.values())
            total_reward = sum(data[2] for data in self.iteration_data.values())
            
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Terminals: {total_terminals}\n")
            
            if total_episodes > 0:
                f.write(f"Terminal Percentage: {(total_terminals / total_episodes * 100):.2f}%\n")
                f.write(f"Average Reward: {(total_reward / total_episodes):.4f}\n")
                
            if self.iteration_data:
                f.write(f"Average Episodes Per Iteration: {(total_episodes / len(self.iteration_data)):.2f}\n")
        
        # Also log final summary to wandb
        if total_episodes > 0:
            wandb.run.summary["final_episodes"] = total_episodes
            wandb.run.summary["final_terminals"] = total_terminals
            wandb.run.summary["final_terminal_percentage"] = (total_terminals / total_episodes * 100)
            wandb.run.summary["final_average_reward"] = (total_reward / total_episodes)

class CustomEnv(MiniWorldEnv):
    """
    Custom Multi-Room Escape Environment for Reinforcement Learning.
    
    This environment implements a 4-room maze where an agent must:
    1. Navigate through rooms by creating doors between them
    2. Reach the hallway (roomD) within a time limit
    3. Eventually reach the terminal goal location
    
    Environment Layout:
    - Room A: Top room (starting area)
    - Room B: Right room 1 (middle-right)
    - Room C: Right room 2 (bottom-right)  
    - Room D: Hallway (left side, connects to terminal)
    - Terminal: Goal location at coordinates [1.0, 20.0]
    
    Key Features:
    - Dynamic door creation when agent touches walls between rooms
    - Comprehensive reward system with multiple components
    - Lidar-based wall distance measurements
    - Room categorization and exploration tracking
    - Configurable reward scaling for different behaviors
    """
    def __init__(self, max_episode_steps, steps_until_hallway,reward_scales=None, **kwargs):
        # Set world dimensions from miniworld_env.py
        self.world_width = 8.0
        self.world_depth = 20.0
        
        # Initialize reward scales with default values if not provided
        self.reward_scales = reward_scales or {
            'reward_orientation_scale': 1.0,
            'reward_distance_scale': 0.0,
            'punishment_distance_scale': 0.0,
            'penalty_stagnation_scale': 1.0,
            'punishment_time_scale': 0.0,
            'reward_hallway_scale': 1.0,
            'reward_connection_scale': 0.0,
            'reward_terminal_scale': 1.0,
            'punishment_terminal_scale': 0.0,
            'punishment_room_scale': 0.0,
            'wall_collision_scale': 1.0
        }
        
        # Terminal is on right side of world
        self.terminal_location = [1.0, 20.0]
        self.max_episode_steps = max_episode_steps
        self.steps_until_hallway = steps_until_hallway  # Maximum steps allowed to reach hallway
        self.has_reached_hallway = False  # Track if agent has reached hallway
        self.hallway_reward_given = False  # Flag to track if hallway reward was given
        self.started_in_hallway = False   # Flag to track if agent started in hallway
        self.non_hallway_steps = 0

        # Placement incrementer for cycling through different starting positions
        self.placement_incrementer = 0
    
        # Add room configurations for random placement
        self.room_configs = [
            # Room A (top room)
            {'min_x': 0.4, 'max_x': 7.6, 'min_z': 0.4, 'max_z': 5.6},
            # Room B (right room 1)
            {'min_x': 2.6, 'max_x': 7.6, 'min_z': 6.6, 'max_z': 12.4},
            # Room C (right room 2)
            {'min_x': 2.6, 'max_x': 7.6, 'min_z': 13.6, 'max_z': 19.6},
        ]

        # Add position stagnation tracking
        self.stagnant_steps = 0  # Counter for steps without movement

        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        self.rooms = []
        self.boxes = []
        self.previous_room = None
        
        # Observation space: comprehensive state representation
        # Includes: agent position, terminal position, distance, directions, lidar, step count, etc.
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0,                   # normalized position of agent
                0, 0,                   # normalized position of terminal
                0,                      # normalized distance to terminal
                -1, -1,                 # normalized agents direction vector
                -1, -1,                 # normalized direction to terminal 
                -1,                     # direction difference
                0, 0, 0, 0, 0,          # lidar measurements (right, left, front, FR, FL)
                0,                      # normalized step-count
                0,                      # normalized stagnation counter
                0,                      # room category
            ], dtype=np.float32),
            high=np.array([
                1, 1,                   # normalized position of agent
                1, 1,                   # normalized position of terminal
                1,                      # normalized distance to terminal
                1, 1,                   # normalized agents direction vector
                1, 1,                   # normalized direction to terminal 
                1,                      # direction difference
                1, 1, 1, 1, 1,          # lidar measurements (right, left, front, FR, FL)
                1,                      # normalized step-count
                1,                      # normalized stagnation counter
                4,                      # room category
            ], dtype=np.float32)
        )

        # Action space: discrete movement actions
        # 0: turn right, 1: turn left, 2: move forward
        self.action_space = spaces.Discrete(3)


    def _gen_world(self):
        """
        Generate the world layout with rooms and initialize environment state.
        
        Creates:
        - 4 rooms with specific dimensions and positions
        - Agent with appropriate size and starting position
        - Connection points between rooms (doors created dynamically)
        - Static world data for rendering
        """
        # Create the agent with appropriate radius for navigation
        self.agent.radius = 0.25

        # Create the hallway along the left side (Room D)
        self.roomD = self.add_rect_room(min_x=0, max_x=2, min_z=6.2, max_z=20)

        # Create the top room (Room A) - main starting area
        self.roomA = self.add_rect_room(min_x=0, max_x=8, min_z=0, max_z=6)

        # Create room 1 on the right side (Room B)
        self.roomB = self.add_rect_room(min_x=2.2, max_x=8, min_z=6.2, max_z=13)

        # Create room 2 on the right side (Room C)
        self.roomC = self.add_rect_room(min_x=2.2, max_x=8, min_z=13.2, max_z=20)

        self.rooms.extend([self.roomA, self.roomB, self.roomC, self.roomD])

        # Place the agent at the default starting position (will be overridden in reset)
        self.place_entity(
            self.agent,
            pos=[4.0, 0, 3.0],  # Center of room A
            dir=0  # Default direction
        )

        # Define room pairs for possible connections
        # Each tuple contains (room1, room2, id1, id2) for connection generation
        self.room_pairs = [
            # Top room and hallway
            (self.roomA, self.roomD, 'A', 'D'),
            # Top room and room right 1
            (self.roomA, self.roomB, 'A', 'B'),
            # Hallway and room right 1
            (self.roomB, self.roomD, 'B', 'D'),
            # Hallway and room right 2
            (self.roomC, self.roomD, 'C', 'D'),
            # Room right 1 and room right 2
            (self.roomB, self.roomC, 'B', 'C')
        ]

        # Add boxes as obstacles (currently disabled - see _add_boxes method)
        #self._add_boxes()

        # Generate connection points (but don't create doors yet - agent will do this)
        self.connections = self.generate_connections(self.room_pairs, [])
        self._gen_static_data()

    
    def _add_boxes(self):
        """
        Add boxes as obstacles in each room to increase navigation complexity.
        
        This method creates strategically placed box obstacles in rooms A, B, and C.
        The boxes are subdivided into smaller units to provide fine-grained obstacle
        detection and navigation challenges.
        
        Note: Currently disabled in _gen_world() but can be enabled for more complex scenarios.
        """
        self.boxes = []
        
        # CONTROL PARAMETERS FOR OVERALL SIZES
        # Room A cluster parameters
        roomA_cluster_width = 0.6      # Width of each box cluster
        roomA_cluster_height = 0.7     # Height of each box cluster
        roomA_cluster_depth = 1.4     # Depth of each box cluster
        
        # Room B and C cluster parameters
        roomBC_cluster_width = 2.5     # Overall width of the box cluster
        roomBC_cluster_height = 0.7    # Height of boxes
        roomBC_cluster_depth = 0.6     # Depth of the box cluster
        
        # Room A - subdivide each box in x and z dimensions
        # Number of subdivisions for Room A boxes
        num_boxes_x_A = 2
        num_boxes_z_A = 4
        
        # Calculate individual box size based on overall cluster size
        small_box_size_A = [
            roomA_cluster_width / num_boxes_x_A,
            roomA_cluster_height,
            roomA_cluster_depth / num_boxes_z_A
        ]
        
        # Base positions for Room A clusters
        boxA_base_positions = [
            [2.5, 0, 1],
            [5.5, 0, 1],
            [2.5, 0, 4],
            [5.5, 0, 4]
        ]
        
        # Generate smaller boxes for each original position in Room A
        for base_pos in boxA_base_positions:
            # Create a grid of smaller boxes
            for i in range(num_boxes_x_A):  # x-direction
                for j in range(num_boxes_z_A):  # z-direction
                    # Calculate offset from base position
                    x_offset = (i - (num_boxes_x_A-1)/2) * small_box_size_A[0]
                    z_offset = (j - (num_boxes_z_A-1)/2) * small_box_size_A[2]
                    
                    pos = [
                        base_pos[0] + x_offset,  # x position with offset
                        base_pos[1],             # y position (ground level)
                        base_pos[2] + z_offset   # z position with offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size_A)
                    box.pos = pos
                    box.dir = 0
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomA)
        
        # For rooms B and C, use grid-based approach
        # Calculate individual box size based on overall cluster dimensions
        num_boxes_x = 4
        num_boxes_z = 2
        
        small_box_size = [
            roomBC_cluster_width / num_boxes_x,
            roomBC_cluster_height,
            roomBC_cluster_depth / num_boxes_z
        ]
        
        # Base positions for clusters in Room B
        boxB_base_positions = [
            [6.2, 0, 8.3],
            [6.2, 0, 11]
        ]
        
        # Generate smaller boxes for Room B
        for base_pos in boxB_base_positions:
            for i in range(num_boxes_x):  # x-direction
                for j in range(num_boxes_z):  # z-direction
                    # Calculate offset from base position
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,  # x position with offset
                        base_pos[1],             # y position (ground level)
                        base_pos[2] + z_offset   # z position with offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomB)
        
        # Base positions for clusters in Room C
        boxC_base_positions = [
            [6.2, 0, 15.3],
            [6.2, 0, 18]
        ]
        
        # Generate smaller boxes for Room C
        for base_pos in boxC_base_positions:
            for i in range(num_boxes_x):  # x-direction
                for j in range(num_boxes_z):  # z-direction
                    # Calculate offset from base position
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,  # x position with offset
                        base_pos[1],             # y position (ground level)
                        base_pos[2] + z_offset   # z position with offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomC)

        

    def generate_connections(self, room_pairs, special_configs):
        """
        Generate possible connection points between rooms for door creation.
        
        This method analyzes room adjacencies and creates potential connection points
        where the agent can later create doors by touching walls.
        
        Args:
            room_pairs: List of tuples defining which rooms can be connected
            special_configs: Dictionary of special connection configurations (currently unused)
            
        Returns:
            Dictionary mapping connection ranges to connection data
        """
        connections = {}
        
        # Convert special_configs to a dictionary if it's a list
        if isinstance(special_configs, list):
            special_configs = {}
        
        def add_connection(room1, room2, id1, id2, is_vertical=False):
            """Add a connection point between two adjacent rooms"""
            # Check for special configuration
            special_config = special_configs.get((id1, id2)) or special_configs.get((id2, id1))
            
            if is_vertical:
                # Vertical adjacency (rooms stacked vertically)
                x_start = round(max(room1.min_x, room2.min_x) + 0.5, 1)
                x_end = round(min(room1.max_x, room2.max_x) - 0.5, 1)
                
                if x_end > x_start:
                    z_top = special_config['z_range'][1] if special_config and 'z_range' in special_config else room1.max_z
                    z_bottom = special_config['z_range'][0] if special_config and 'z_range' in special_config else room2.min_z
                    
                    connections[((x_start, x_end), 
                            (round(z_top - 0.1, 1), z_top))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((x_start, x_end),
                            (z_bottom, round(z_bottom + 0.1, 1)))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
            else:
                # Horizontal adjacency (rooms side by side)
                z_start = round(max(room1.min_z, room2.min_z) + 0.5, 1)
                z_end = round(min(room1.max_z, room2.max_z) - 0.5, 1)
                
                if special_config and 'z_range' in special_config:
                    z_start, z_end = special_config['z_range']
                
                if z_end > z_start:
                    connections[((round(room1.max_x - 0.1, 1), room1.max_x),
                            (z_start, z_end))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((room2.min_x, round(room2.min_x + 0.1, 1)),
                            (z_start, z_end))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
        
        # Process each room pair to find adjacencies and create connection points
        for room1, room2, id1, id2 in room_pairs:
            # Check adjacency in x-direction (horizontal)
            if abs(room1.max_x - room2.min_x) < 0.3:
                add_connection(room1, room2, id1, id2)
            elif abs(room2.max_x - room1.min_x) < 0.3:
                add_connection(room2, room1, id2, id1)
                
            # Check adjacency in z-direction (vertical)
            if abs(room1.max_z - room2.min_z) < 0.3:
                add_connection(room1, room2, id1, id2, True)
            elif abs(room2.max_z - room1.min_z) < 0.3:
                add_connection(room2, room1, id2, id1, True)
                    
        return connections
    
    def _get_current_room(self):
        """
        Determine which room the agent is currently in.
        
        Uses the agent's position to determine room occupancy with boundary tolerance
        to handle edge cases where the agent might be near room boundaries.
        
        Returns:
            String identifier of the current room (e.g., 'roomA', 'roomB', etc.)
        """
        agent_pos = self.agent.pos
        boundary_tolerance = 0.2  # Increased tolerance for room boundaries
        
        # Check each room
        for room in self.rooms:
            if ((room.min_x - boundary_tolerance <= agent_pos[0] <= room.max_x + boundary_tolerance) and 
                (room.min_z - boundary_tolerance <= agent_pos[2] <= room.max_z + boundary_tolerance)):
                
                # Find the room's name
                for attr_name, attr_value in vars(self).items():
                    if attr_name.startswith('room') and attr_value == room:
                        return attr_name
        
        # If no room is found but we have a previous room, return that
        if hasattr(self, 'previous_room'):
            return self.previous_room
                
        return "unknown"
    
    def _get_room_category(self, room_name):
        """
        Get numerical room category for observation space.
        
        Categories:
        0: Hallway (roomD) - the key room that leads to terminal
        1-3: Other rooms (roomA, roomB, roomC)
        4: Unknown room
        
        Args:
            room_name: String name of the room
            
        Returns:
            Integer category code for the room
        """
        if room_name == 'roomD':
            return 0
        elif room_name == 'roomA':
            return 1
        elif room_name == 'roomB':
            return 2
        elif room_name == 'roomC':
            return 3
        else:
            return 4

    def get_wall_distance(self, pos, direction, max_distance=10):
        """
        Cast a ray from position in given direction and return distance to nearest wall.
        
        This method implements a simple raycasting algorithm to measure distances
        to walls and obstacles, providing lidar-like sensor data for the agent.
        
        Args:
            pos: Starting position for the ray
            direction: Direction vector for the ray
            max_distance: Maximum distance to check
            
        Returns:
            Distance to the nearest wall or obstacle
        """
        step_size = 0.1
        current_pos = np.array(pos)
        
        for i in range(int(max_distance / step_size)):
            current_pos = current_pos + direction * step_size
            
            # Check if position is inside any room
            in_room = False
            for room in self.rooms:
                if (room.min_x <= current_pos[0] <= room.max_x and
                    room.min_z <= current_pos[2] <= room.max_z):
                    in_room = True
                    break
            
            if not in_room:
                # Hit a wall - return distance traveled
                return i * step_size
                
        return max_distance
    
    def get_lidar_measurements(self):
        """
        Get distances to walls in different directions around the agent.
        
        Provides 5-directional distance measurements:
        - Forward: agent's current facing direction
        - Left/Right: perpendicular to forward direction
        - Forward-left/Forward-right: diagonal measurements
        
        Returns:
            Tuple of (right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist)
        """
        # Forward direction is agent's current direction
        forward_dir = self.agent.dir_vec
        
        # Right is rotated 90 degrees from forward (clockwise)
        right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])
        
        # Left is rotated -90 degrees from forward (counterclockwise)
        left_dir = np.array([forward_dir[2], 0, -forward_dir[0]])
        
        # Diagonal directions
        forward_right_dir = forward_dir + right_dir
        forward_right_dir = forward_right_dir / np.linalg.norm(forward_right_dir)
        
        forward_left_dir = forward_dir + left_dir
        forward_left_dir = forward_left_dir / np.linalg.norm(forward_left_dir)
        
        # Get distances
        forward_dist = self.get_wall_distance(self.agent.pos, forward_dir)
        left_dist = self.get_wall_distance(self.agent.pos, left_dir)
        right_dist = self.get_wall_distance(self.agent.pos, right_dir)
        forward_right_dist = self.get_wall_distance(self.agent.pos, forward_right_dir)
        forward_left_dist = self.get_wall_distance(self.agent.pos, forward_left_dir)
        
        return right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist
    
    def _normalize_terminal_distance(self, distance):
        """
        Normalize distance to terminal for observation space.
        
        Converts raw distance to a normalized value where higher values
        indicate closer proximity to the terminal (better for the agent).
        
        Args:
            distance: Raw distance to terminal
            
        Returns:
            Normalized distance value (0-1, where 1 is closest)
        """
        max_dist = np.sqrt(self.world_width**2 + self.world_depth**2)  # Max possible distance
        return (max_dist - distance) / max_dist

    def _agent_touches_wall(self, agent_nose_pos):
        """
        Check if the agent is touching a wall between rooms for door creation.
        
        Determines if the agent's position (specifically the nose/front of the agent)
        is close enough to a wall connection point to trigger door creation.
        
        Args:
            agent_nose_pos: Position of the agent's front/nose
            
        Returns:
            Connection key if touching a wall, None otherwise
        """
        touching_threshold = 0.1  # Threshold for wall touches
        for (x_range, z_range), data in self.connections.items():
            if (x_range[0]-touching_threshold <= agent_nose_pos[0] <= x_range[1]+touching_threshold and 
                z_range[0]-touching_threshold <= agent_nose_pos[2] <= z_range[1]+touching_threshold):
                return (x_range, z_range)
        
        return None
    
    def _create_doors(self, new_connection):
        """
        Create a physical door at a connection point between rooms.
        
        This method is called when the agent touches a wall and creates an actual
        passageway between two rooms, allowing navigation between them.
        
        Args:
            new_connection: The connection key identifying which wall to open
        """
        if not hasattr(self, 'special_connection_groups'):
            self.special_connection_groups = {}

        x_range, z_range = new_connection
        connection = self.connections[new_connection]['connection']
        room1_id, room2_id = connection.split('-')
        
        room1_attr = f"room{room1_id}"
        room2_attr = f"room{room2_id}"
        
        room1 = getattr(self, room1_attr)
        room2 = getattr(self, room2_attr)
        
        door_width = 1.0
        min_offset = 0.05
        
        # Create the physical door based on room adjacency type
        if abs(room1.max_x - room2.min_x) < 0.3 or abs(room2.max_x - room1.min_x) < 0.3:
            # Horizontal adjacency (side by side)
            left_room = room1 if room1.max_x < room2.min_x else room2
            right_room = room2 if left_room == room1 else room1
            
            min_z = self.agent.pos[2] - door_width/2
            min_z = np.clip(min_z,
                        max(left_room.min_z, right_room.min_z) + min_offset,
                        min(left_room.max_z, right_room.max_z) - door_width - min_offset)
            
            self.connect_rooms(left_room, right_room, min_z=min_z, max_z=min_z + door_width)
        else:
            # Vertical adjacency (one above the other)
            top_room = room1 if room1.max_z > room2.max_z else room2
            bottom_room = room2 if top_room == room1 else room1
            
            min_x = self.agent.pos[0] - door_width/2
            min_x = np.clip(min_x,
                        max(top_room.min_x, bottom_room.min_x) + min_offset,
                        min(top_room.max_x, bottom_room.max_x) - door_width - min_offset)
            
            self.connect_rooms(bottom_room, top_room, min_x=min_x, max_x=min_x + door_width)
        
        # Update connection states
        current_connection = f"{room1_id}-{room2_id}"
        reverse_connection = f"{room2_id}-{room1_id}"
        
        # Store the last created connection for tracking
        self.last_created_connection = {
            "from": room1_id,
            "to": room2_id,
            "connection_str": current_connection
        }
        
        # Check if this is part of a special group
        special_group = None
        if hasattr(self, 'special_connection_groups'):
            for group_key, connections in self.special_connection_groups.items():
                if current_connection in connections or reverse_connection in connections:
                    special_group = group_key
                    break
        
        # Update connection states based on whether it's a special group
        if special_group:
            # Mark all connections in the group as created
            group_connections = self.special_connection_groups[special_group]
            for key, value in self.connections.items():
                if value['connection'] in group_connections:
                    self.connections[key]['created'] = True
        else:
            # Regular connection - only mark this connection and its reverse
            for key, value in self.connections.items():
                if value['connection'] in [current_connection, reverse_connection]:
                    self.connections[key]['created'] = True
        

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        This method initializes a new episode by:
        - Resetting all tracking variables
        - Setting up the world with random or predefined agent placement
        - Resetting door connections
        - Calculating initial observations
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
            
        Returns:
            Tuple of (observation_array, info_dict)
        """
        # Reset tracking variables
        self.has_reached_hallway = False
        self.hallway_reward_given = False
        self.stagnant_steps = 0
        self.non_hallway_steps = 0 
        self.reached_terminal_area = False
        
        # Set seed if provided, otherwise use a random one
        if seed is not None:
            self.seed(seed)
        else:
            # Generate a new random seed if none provided
            new_seed = int(time.time() * 1000) % 100000
            self.seed(new_seed)

        # Define predefined positions for each room [pos, direction]
        if not hasattr(self, 'all_positions'):
            # Room A positions (indices 0-3)
            # Room B positions (indices 4-6) 
            # Room C positions (indices 7-9)
            self.all_positions = [
                # Room A (top room)
                ([0.75, 0, 1], -np.pi/2),
                ([4, 0, 1], -np.pi/2),
                ([7.25, 0, 1], -np.pi/2),
                ([7.25, 0, 5], np.pi),
                # Room B (right room 1)
                ([7.25, 0, 7], np.pi),
                ([7.25, 0, 9.5], np.pi),
                ([7.25, 0, 12], np.pi),
                # Room C (right room 2)
                ([7.25, 0, 14], np.pi),
                ([7.25, 0, 16.5], np.pi),
                ([7.25, 0, 19.2], np.pi),
            ]
            
            # Define position ranges for each room
            self.room_position_ranges = {
                0: (0, 4),    # Room A: positions 0-3
                1: (4, 7),    # Room B: positions 4-6
                2: (7, 10)    # Room C: positions 7-9
            }
        
        # Determine which room to place the agent in (cycle through rooms A, B, C)
        room_index = (self.placement_incrementer // 3) % len(self.room_configs)
        self.placement_incrementer += 1
        
        # Decide whether to use a predefined position (4/5 chance) or random position (1/5 chance)
        use_predefined = random.random() < 4/5
        
        # Call parent reset to create the world
        observation = super().reset(seed=self.seed_value)
        
        if use_predefined:
            # Use a predefined position from the appropriate room
            start, end = self.room_position_ranges[room_index]
            position_index = random.randrange(start, end)
            pos, direction = self.all_positions[position_index]
            
            # Place the agent at the predefined position and direction
            self.place_entity(
                self.agent,
                pos=pos,
                dir=direction
            )
        else:
            # Get the corresponding room object for random placement
            if room_index == 0:
                room = self.roomA
            elif room_index == 1:
                room = self.roomB
            else:
                room = self.roomC
            
            # Use random placement within the room
            self.place_entity(
                self.agent,
                room=room,
                dir=random.uniform(-np.pi, np.pi)
            )
        
        # Reset the connections to False (no doors exist yet)
        for connection in self.connections:
            self.connections[connection]['created'] = False

        # Check if agent started in hallway
        current_room = self._get_current_room()
        self.started_in_hallway = current_room == 'roomD'
        self.previous_room = current_room

        if self.started_in_hallway:
            self.has_reached_hallway = True
            self.hallway_reward_given = True  # Don't give hallway reward if starting in hallway

        # Calculate initial distance to terminal
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        self.previous_distance_to_terminal = distance_to_terminal

        # Get observation array
        observation_array = self._get_observation_array()
        
        return observation_array, {}

    def seed(self, seed=None):
        """
        Seed the environment's random number generator for reproducibility.
        
        Args:
            seed: Random seed value
            
        Returns:
            List containing the seed value
        """
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_observation_array(self):
        """
        Create the comprehensive observation array for the agent.
        
        The observation includes:
        - Normalized agent and terminal positions
        - Distance to terminal
        - Direction vectors and alignment measures
        - Lidar measurements for obstacle detection
        - Step count and stagnation tracking
        - Current room category
        
        Returns:
            Normalized numpy array representing the current state
        """
        # Normalize agent's position
        norm_agent_pos = np.array([
        self.agent.pos[0] / self.world_width,
        0,  # Skip y-coordinate normalization
        self.agent.pos[2] / self.world_depth
        ])
        
        # Normalize to [0,1] range
        norm_agent_pos[0] = (norm_agent_pos[0] + 1) / 2  # Normalize x-coordinate to [0,1]
        norm_agent_pos[2] = (norm_agent_pos[2] + 1) / 2  # Normalize z-coordinate to [0,1]

        # Normalize terminal position
        norm_terminal_pos = np.array(self.terminal_location) / np.array([self.world_width, self.world_depth])
        norm_terminal_pos[0] = (norm_terminal_pos[0] + 1) / 2  # Normalize x-coordinate to [0,1]
        norm_terminal_pos[1] = (norm_terminal_pos[1] + 1) / 2  # Normalize z-coordinate to [0,1]

        # Calculate direction vectors
        # Vector from agent to terminal
        dx = self.terminal_location[0] - self.agent.pos[0]
        dz = self.terminal_location[1] - self.agent.pos[2]
        
        # Direction vector to terminal (with flipped z for correct orientation)
        direction_vector = np.array([dx, -dz])
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length  # Normalize
        else:
            direction_vector = np.array([1.0, 0.0])  # Default forward if at same position
        
        # Agent's forward direction vector (always normalized)
        agent_dir_vec = np.array([
            np.cos(self.agent.dir),
            np.sin(self.agent.dir)
        ])
        
        # Calculate angle between vectors for reward purposes
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Determine if angle is positive or negative (for turning direction)
        cross_z = agent_dir_vec[0] * direction_vector[1] - agent_dir_vec[1] * direction_vector[0]
        if cross_z < 0:
            angle_difference = -angle_difference
        
        # Store for use in step() method
        self.last_angle_difference = angle_difference
        self.last_dot_product = dot_product

        # Calculate distance to terminal
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        
        # Normalize terminal distance
        norm_dist_term = self._normalize_terminal_distance(distance_to_terminal)

        # Get lidar measurements
        right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()

        # Normalize lidar distances
        max_lidar_dist = 10.0
        norm_right = right_dist / max_lidar_dist
        norm_left = left_dist / max_lidar_dist
        norm_forward = forward_dist / max_lidar_dist
        norm_forward_right = forward_right_dist / max_lidar_dist
        norm_forward_left = forward_left_dist / max_lidar_dist

        # Get current room and category
        current_room = self._get_current_room()
        room_category = self._get_room_category(current_room)
        
        # Build observation array with normalized vectors
        return np.array([
            norm_agent_pos[0], norm_agent_pos[2],       # normalized position of agent (2)
            norm_terminal_pos[0], norm_terminal_pos[1], # normalized position of terminal (2)
            norm_dist_term,                             # normalized distance to terminal (1)
            agent_dir_vec[0], agent_dir_vec[1],         # agent direction vector components (2)
            direction_vector[0], direction_vector[1],   # direction to terminal vector components (2)
            dot_product,                                # dot product (alignment measure) (1)
            norm_right, norm_left, norm_forward,        # lidar measurements (3)
            norm_forward_right, norm_forward_left,      # diagonal lidar measurements (2)
            self.step_count / self.max_episode_steps,   # normalized step count (1)
            self.stagnant_steps / 100,                  # normalized stagnation counter (1)
            room_category,                              # room category (1)
        ], dtype=np.float32)

    def step(self, action):
        """
        Execute an action and return the new state, reward, and episode status.
        
        This is the main environment step function that:
        1. Processes the agent's action
        2. Updates the environment state
        3. Calculates rewards based on multiple components
        4. Checks for episode termination conditions
        5. Returns the new observation and info
        
        Args:
            action: Integer action (0=turn right, 1=turn left, 2=move forward)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous agent position for movement detection
        previous_agent_pos = np.array(self.agent.pos)
    
        # Call parent step method to execute the action
        observation, reward, terminated, truncated, info = super().step(action)

        # Initialize reward components (will be scaled later)
        reward_orientation = 0
        reward_distance_terminal = 0
        punishment_distance_terminal = 0
        punishment_time = 0
        reward_hallway = 0
        reward_connection = 0
        reward_terminal = 0
        punishment_terminal = 0
        punishment_room = 0
        penalty_stagnation = 0
        wall_collision_penalty = 0

        # Get agent's nose position (for wall touch detection)
        agent_nose_pos = self.agent.pos + self.agent.dir_vec * self.agent.radius
        current_agent_pos = np.array(self.agent.pos)
        
        # Calculate distance to terminal
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )

        # Calculate direction vectors for orientation rewards
        # Vector from agent to terminal
        direction_vector = np.array([
            self.terminal_location[0] - self.agent.pos[0],
            -(self.terminal_location[1] - self.agent.pos[2])  # FLIPPED z-component
        ])
        
        # Normalize direction vector
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        
        # Agent's forward direction vector
        agent_dir_vec = np.array([
            np.cos(self.agent.dir),
            np.sin(self.agent.dir)
        ])
        
        # Calculate dot product and angle difference
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Optional debug printing
        if hasattr(self, 'debug') and self.debug:
            print("direction vector:", direction_vector)
            print("agent direction vector:", agent_dir_vec)
            print("dot product:", dot_product)
            print("angle difference:", angle_difference)

        last_angle_difference = self.last_angle_difference
        
        # Add time punishment for each step to encourage efficiency
        punishment_time -= 0.5

        # Check if agent moved significantly
        position_changed = np.linalg.norm(current_agent_pos - previous_agent_pos) > 0.1

        # Track stagnant steps (when agent doesn't move)
        if position_changed:
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        # Apply stagnation punishment if agent is stuck too long
        if self.stagnant_steps >= 100:
            penalty_stagnation = -100
            info['penalty_stagnation'] = penalty_stagnation
            info['stagnation_penalty_applied'] = True
            self.stagnant_steps = 0

        # Apply orientation reward based on alignment with terminal direction
        # Punish if agent is not facing roughly toward the terminal
        if dot_product < np.cos(np.pi / 9):  # Within 20 degrees
            reward_orientation -= 0.1
        
        # Update previous distance if position changed
        if position_changed:
            self.previous_distance_to_terminal = distance_to_terminal

        # Get current room and check if agent is in hallway
        current_room = self._get_current_room()
        is_in_hallway = current_room == 'roomD'
        
        if is_in_hallway:
            # Small constant reward for being in hallway (key room)
            reward_hallway += 0.05
            self.non_hallway_steps = 0  # Reset counter when in hallway
            
            # Check if this is the first time reaching hallway this episode
            first_hallway_visit = not self.has_reached_hallway
            
            # Only give the one-time large reward if ALL conditions are met
            if (first_hallway_visit and          # Must be first time reaching hallway this episode
                not self.started_in_hallway and  # Must not have started in hallway
                not self.hallway_reward_given):  # Must not have already given the reward
                
                # Large one-time reward could be added here if desired
                #reward_hallway += 100  # Large one-time reward
                self.hallway_reward_given = True
            
            # Always set this after being in hallway
            self.has_reached_hallway = True
        else:
            # Track time spent outside hallway
            self.non_hallway_steps += 1
            
            # Larger punishment if spending too long outside hallway
            if self.non_hallway_steps >= 100:
                #punishment_room -= 100  # Could add punishment here
                self.non_hallway_steps = 0  # Reset counter after applying penalty

        # Check if hallway timeout has been reached
        if not self.has_reached_hallway and self.step_count >= self.steps_until_hallway:
            truncated = True
            punishment_terminal = -500  # Apply punishment for not reaching hallway in time
            info['punishment_terminal'] = punishment_terminal

        # Handle agent touching walls and creating doors
        new_connection = self._agent_touches_wall(agent_nose_pos)
        if new_connection is not None and not self.connections[new_connection]['created']:
            self._create_doors(new_connection)
            #reward_connection += 10  # Could add reward for door creation
            self._gen_static_data()
            # Force rendering update after creating the portal
            self._render_static()
            info['created_door'] = True
        
            # Add connection information to info for logging
            connection_info = self.connections[new_connection]['connection'].split('-')
            info['door_connection_from'] = connection_info[0]
            info['door_connection_to'] = connection_info[1]

        # Check for wall collisions (when not creating doors)
        wall_collision_penalty = 0
        collision_threshold = 0.5  # Threshold for detecting wall/box proximity
        is_touching_connection = new_connection is not None

        # Only apply wall collision penalty if not at a connection point
        if not is_touching_connection:
            # Check if any lidar measurement is below threshold
            right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()
            
            if (forward_dist < collision_threshold or 
                right_dist < collision_threshold or 
                left_dist < collision_threshold or
                forward_right_dist < collision_threshold or
                forward_left_dist < collision_threshold):
                
                # Calculate penalty based on how close we are to the wall or box
                closest_dist = min(forward_dist, right_dist, left_dist, forward_right_dist, forward_left_dist)
                wall_collision_penalty = -1 * (1 - (closest_dist / collision_threshold))
                
                # Add to info for logging
                info['collision'] = True
                info['collision_penalty'] = wall_collision_penalty

        # Check if agent has not reached hallway by deadline
        if not self.has_reached_hallway and self.step_count >= self.steps_until_hallway:
            truncated = True
            info['truncation_reason'] = 'hallway_timeout'
            punishment_terminal = -100
            info['punishment_terminal'] = punishment_terminal

        # Check if agent has reached terminal location (main goal)
        terminal_x, terminal_z = self.terminal_location
        if (terminal_x - 0.2 <= agent_nose_pos[0] <= terminal_x + 0.2 and 
            terminal_z - 0.2 <= agent_nose_pos[2] <= terminal_z + 0.2):
            terminated = True
            self.reached_terminal_area = True
            
            # Large reward for reaching terminal, with possible step-based scaling
            reward_terminal = 500
            
            info['reached_terminal'] = True

        # Handle truncation at max episode steps
        if self.step_count >= self.max_episode_steps:
            truncated = True
            punishment_terminal -= 100
            info['max_steps_reached'] = True

        # Store raw reward components for logging
        raw_reward_components = {
            'reward_orientation': reward_orientation,
            'reward_distance': reward_distance_terminal,
            'punishment_distance': punishment_distance_terminal,
            'penalty_stagnation': penalty_stagnation,
            'punishment_time': punishment_time,
            'reward_hallway': reward_hallway,
            'reward_connection': reward_connection,
            'reward_terminal': reward_terminal,
            'punishment_terminal': punishment_terminal,
            'punishment_room': punishment_room,
            'wall_collision_penalty': wall_collision_penalty
        }
        
        # Apply reward scales to calculate total reward
        reward = (
            self.reward_scales['reward_orientation_scale'] * reward_orientation +
            self.reward_scales['reward_distance_scale'] * reward_distance_terminal +
            self.reward_scales['punishment_distance_scale'] * punishment_distance_terminal +
            self.reward_scales['penalty_stagnation_scale'] * penalty_stagnation +
            self.reward_scales['punishment_time_scale'] * punishment_time +
            self.reward_scales['reward_hallway_scale'] * reward_hallway +
            self.reward_scales['reward_connection_scale'] * reward_connection +
            self.reward_scales['reward_terminal_scale'] * reward_terminal +
            self.reward_scales['punishment_terminal_scale'] * punishment_terminal +
            self.reward_scales['punishment_room_scale'] * punishment_room +
            self.reward_scales['wall_collision_scale'] * wall_collision_penalty 
        )

        # Update info dictionary with reward components and episode data
        info.update(raw_reward_components)
        info.update({
            'episode_rewards': reward,
            'step_count': self.step_count,
            'current_room': current_room
        })

        # Get observation array for return
        observation_array = self._get_observation_array()
        
        return observation_array, reward, terminated, truncated, info


class RobustNoRenderEnv(CustomEnv):
    """
    A version of the environment that doesn't attempt to render graphics.
    
    This class is designed for training on headless servers or systems without
    graphics capabilities. It disables all rendering operations while maintaining
    full environment functionality.
    
    Useful for:
    - Training on cloud servers without GPU/graphics support
    - Batch training jobs
    - Environments where rendering would slow down training
    """
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        # Set minimal window size to avoid graphics issues
        kwargs.update({
            'window_width': 1,
            'window_height': 1
        })
        super().__init__(max_episode_steps=max_episode_steps, steps_until_hallway=steps_until_hallway,reward_scales=reward_scales, **kwargs)
        self.render_mode = 'none'
        
    def render_obs(self, vis_fb=None):
        """Disable observation rendering by returning empty array"""
        return np.zeros((3, 84, 84), dtype=np.uint8)
    
    def _render_static(self):
        """Disable static rendering"""
        pass
        
    def _render_dynamic(self):
        """Disable dynamic rendering"""
        pass


class CheckpointCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints during training.
    
    Saves model checkpoints at regular intervals based on training iterations,
    allowing for recovery and analysis of training progress.
    """
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.iteration_count = 0

    def _on_step(self) -> bool:
        """Check if it's time to save a checkpoint"""
        if not hasattr(wandb, 'config') or 'iterations' not in wandb.config:
            return True
            
        # Calculate current iteration based on timesteps
        timesteps_per_iteration = self.locals.get('total_timesteps') // wandb.config['iterations']
        current_iteration = self.num_timesteps // timesteps_per_iteration

        # Save if we've reached a new 10th iteration
        if current_iteration > self.iteration_count and current_iteration % 10 == 0:
            self.iteration_count = current_iteration
            path = f"{self.save_path}/{self.name_prefix}_iteration_{current_iteration}"
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint at iteration {current_iteration}")
        return True


def make_env(max_episode_steps, steps_until_hallway, seed, reward_scales=None):
    """
    Environment factory function for creating vectorized environments.
    
    This function creates a single environment instance with the specified
    parameters and wraps it with monitoring capabilities.
    
    Args:
        max_episode_steps: Maximum steps per episode
        steps_until_hallway: Maximum steps to reach hallway
        seed: Random seed for this environment
        reward_scales: Dictionary of reward scaling factors
        
    Returns:
        Function that creates the environment when called
    """
    def _init():
        try:
            env = RobustNoRenderEnv(max_episode_steps=max_episode_steps, steps_until_hallway=steps_until_hallway, reward_scales=reward_scales)
            env = Monitor(env)
            env.unwrapped.seed(seed)
            return env
        except Exception as e:
            print(f"Error initializing environment with seed {seed}:")
            print(traceback.format_exc())
            raise
    return _init


def run_experiment(config):
    """
    Run a complete training experiment with the given configuration.
    
    This function:
    1. Sets up the environment and training infrastructure
    2. Configures callbacks for logging and checkpointing
    3. Trains the PPO agent
    4. Saves results and cleans up resources
    
    Args:
        config: Dictionary containing all experiment parameters
    """
    # Update config with additional training parameters
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": config['timesteps'] * config['iterations'],
        "env_name": "CustomEnv-EscapeRoom-v1",
        **config
    }
    
    # Define experiment name for tracking
    # v4 = fixed wall collision and lidar detection, also includes boxes (currently disabled)
    # + made boxes smaller for better maneuvering with the bounding box issue
    run_name = "escape-room-v4_noboxes"
    
    # Initialize Weights & Biases for experiment tracking - use your own project name!
    with wandb.init(project="simple_multi-single",
                    name=run_name,
                    config=config,
                    sync_tensorboard=True) as run:
        # Save the training script for reproducibility
        wandb.save("env_sarl_v1.py")
        
        # Create vectorized environments with unique seeds
        num_envs = 1
        base_seed = config['seed']
        
        # Generate unique seeds for each environment to ensure diversity
        env_seeds = [base_seed + i * 1000 for i in range(num_envs)]
        
        # Create vectorized environment
        envs = DummyVecEnv([
            make_env(
                config['max_episode_steps'], 
                config['steps_until_hallway'], 
                env_seeds[i],  # Use unique seed for each env
                {k: config[k] for k in config if k.endswith('_scale')}
            ) for i in range(num_envs)
        ])
        
        # Create PPO model with specified parameters
        model = PPO("MlpPolicy",
                    envs,
                    verbose=1,
                    tensorboard_log=f"runs/{run_name}_{wandb.run.id}",
                    seed=base_seed,
                    device="auto")
        

        # Create directories for logging and checkpoints
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        tracking_dir = f"tracking_logs/{wandb.run.id}"
        tracking_light_dir = f"tracking_doors_logs/{wandb.run.id}"
        script_dir = f"script_logs/{wandb.run.id}"
        os.makedirs(script_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tracking_dir, exist_ok=True)
        os.makedirs(tracking_light_dir, exist_ok=True)
        
        # Save a backup copy of the training script
        script_path = os.path.abspath(__file__)
        script_backup_path = os.path.join(script_dir, "train_backup.py")
        try:
            shutil.copy2(script_path, script_backup_path)
            print(f"Saved a backup of the training script to: {script_backup_path}")
        except Exception as e:
            print(f"Failed to save script backup: {e}")

        # Set up enhanced reward tracking callback
        enhanced_reward_callback = EnhancedRewardCallback(
            log_dir=f"episode_logs/{wandb.run.id}",
            timesteps_per_iteration=config['timesteps'],
            verbose=1
        )

        # Set up all training callbacks
        callbacks = [
            # Weights & Biases integration for metric logging
            WandbCallback(
                model_save_path=f"models/{wandb.run.id}_PPO_{run_name}",
                verbose=2,
            ),
            # Enhanced episode and reward tracking
            enhanced_reward_callback,
            # Model checkpoint saving
            CheckpointCallback(
                save_freq=1,
                save_path=checkpoint_dir,
                name_prefix=f"checkpoint_{wandb.run.id}",
                verbose=1
            ),
            # Detailed behavior tracking
            TrackerCallback(log_dir=tracking_dir, verbose=1),
            # Simplified door creation and terminal tracking
            ImprovedTrackerCallback(log_dir=tracking_light_dir, verbose=1),
        ]
        
        try:
            # Train the model
            total_timesteps = config['timesteps'] * config['iterations']
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name='PPO_simple-match_'
            )
            # Save final model
            model.save(f"{checkpoint_dir}/final_model.zip")
        finally:
            # Clean up resources
            envs.close()
            del model
            gc.collect()


def main():
    """
    Main function to run the training experiment.
    
    Sets up multiprocessing, generates a random seed, and runs the experiment
    with the default configuration parameters.
    """
    try:
        # Set multiprocessing start method for compatibility
        multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        pass

    # Generate random seed based on current time for experiment uniqueness
    random_seed = int(time.time()) % 10000
    
    # Default configuration with comprehensive reward scaling options
    config = {
        'seed': random_seed,  # Use random seed instead of hardcoded value
        'max_episode_steps': 500,  # Maximum steps per episode before truncation
        'timesteps': 12000,  # Timesteps per training iteration
        'iterations': 1000,  # Number of training iterations
        'steps_until_hallway': 500,  # Maximum steps to reach hallway before penalty
        
        # Reward scaling factors - these control the relative importance of different behaviors
        'reward_orientation_scale': 1.0,    # Reward for facing toward terminal
        'reward_distance_scale': 0.0,       # Reward for moving closer to terminal
        'punishment_distance_scale': 0.0,   # Punishment for moving away from terminal
        'penalty_stagnation_scale': 1.0,    # Penalty for not moving (being stuck)
        'punishment_time_scale': 0.0,       # Penalty applied each step (time pressure)
        'reward_hallway_scale': 1.0,        # Reward for being in/reaching hallway
        'reward_connection_scale': 0.0,     # Reward for creating doors between rooms
        'reward_terminal_scale': 1.0,       # Large reward for reaching terminal goal
        'punishment_terminal_scale': 0.0,   # Punishment for failing to reach terminal
        'punishment_room_scale': 0.0,       # Punishment for being outside hallway
        'wall_collision_scale': 1.0         # Penalty for colliding with walls/obstacles
    }
    
    # Log the seed being used for this experiment
    print(f"Running experiment with random seed: {random_seed}")
    
    # Start the training experiment
    run_experiment(config)


if __name__ == "__main__":
    """
    Entry point for the training script.
    
    This script implements a comprehensive reinforcement learning environment
    for training agents to navigate a multi-room maze. The agent must learn to:
    
    1. Navigate through rooms by creating doors between them
    2. Reach a hallway within a time limit  
    3. Eventually reach the terminal goal location
    
    Key Features:
    - Multi-room environment with dynamic door creation
    - Comprehensive reward system with configurable scaling
    - Extensive logging and tracking for analysis
    - Robust training infrastructure with checkpointing
    - Integration with Weights & Biases for experiment tracking
    
    The environment is designed to test exploration, navigation, and goal-directed
    behavior in a structured but challenging setting.
    """
    main()