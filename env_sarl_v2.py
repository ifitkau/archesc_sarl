# Single-Agent Escape Room Training Environment
# =============================================
# This script implements a single-agent reinforcement learning environment using Stable Baselines3
# and PPO algorithm. The agent learns to navigate through rooms, create doors, and reach a terminal
# location while optimizing various reward components.

# Standard library imports for environment simulation and data handling
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import os
import shutil
import sys
import random
import multiprocessing
import gc
import traceback
import itertools

# MiniWorld environment components for 3D simulation
from miniworld.entity import Agent, Box, COLORS
from miniworld.miniworld import MiniWorldEnv, Room

# Stable Baselines3 imports for reinforcement learning
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback

# External libraries for experiment tracking and data visualization
from wandb.integration.sb3 import WandbCallback
import wandb
from tabulate import tabulate

# Custom Callback Classes for Training Analysis
# =============================================
# These callbacks track various aspects of training for analysis and debugging

class ImprovedTrackerCallback(BaseCallback):
    """
    Advanced callback that tracks door creation positions and episode outcomes in CSV format.
    Creates separate files for each environment in vectorized training scenarios.
    This provides detailed analysis of agent behavior and door placement strategies.
    """
    def __init__(self, log_dir="./logs", verbose=0):
        super(ImprovedTrackerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Track data separately for each environment in vectorized setups
        self.env_data = {}           # Episode and step tracking per environment
        self.log_files = {}          # Text log files for human-readable output
        self.door_csv_files = {}     # CSV files for door creation data
        self.terminal_csv_files = {} # CSV files for terminal reaching events
        self.episode_outcomes_files = {} # CSV files for episode outcome summaries
        self.timestamp = int(time.time())  # Unique timestamp for file naming
    
    def _on_training_start(self):
        """
        Initialize CSV files with appropriate headers at the start of training.
        Creates structured data files for post-training analysis.
        """
        # Create separate tracking files for each environment
        for env_id in range(self.model.get_env().num_envs):
            # Door creation events tracking
            door_csv_path = f"{self.log_dir}/door_creations_env{env_id}_{self.timestamp}.csv"
            self.door_csv_files[env_id] = door_csv_path
            
            # Terminal reaching events tracking
            terminal_csv_path = f"{self.log_dir}/terminal_events_env{env_id}_{self.timestamp}.csv"
            self.terminal_csv_files[env_id] = terminal_csv_path
            
            # Episode outcome summaries
            episode_outcomes_path = f"{self.log_dir}/episode_outcomes_env{env_id}_{self.timestamp}.csv"
            self.episode_outcomes_files[env_id] = episode_outcomes_path
            
            # Initialize CSV files with headers
            with open(door_csv_path, "w") as f:
                # Include terminal_reached to correlate door creation with episode success
                f.write("episode,step,x,y,z,room,from_room,to_room,is_door,terminal_reached\n")
                
            with open(terminal_csv_path, "w") as f:
                f.write("episode,step,x,y,z,room,reward,total_steps\n")
                
            with open(episode_outcomes_path, "w") as f:
                f.write("episode,terminal_reached,steps,rooms_visited\n")
    
    def _on_step(self) -> bool:
        """
        Called at each environment step to track agent behavior and door creation events.
        Processes information from all environments in vectorized training.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Process each environment in the vectorized environment
        for i, info in enumerate(self.locals['infos']):
            env_id = i  # Environment identifier in vectorized setup
            
            # Initialize tracking data for new environments
            if env_id not in self.env_data:
                self.env_data[env_id] = {
                    "episode_count": 0,           # Total episodes completed
                    "current_episode": 0,         # Current episode number
                    "terminal_reached": False,    # Whether terminal was reached this episode
                    "rooms_visited": set(),       # Set of rooms visited this episode
                    "current_episode_step": 0     # Steps taken in current episode
                }
                
                # Initialize text log file for human-readable tracking
                self.log_files[env_id] = f"{self.log_dir}/tracking_env{env_id}_{self.timestamp}.txt"
                if not os.path.exists(self.log_files[env_id]):
                    with open(self.log_files[env_id], "w") as f:
                        f.write(f"=== DOOR AND REWARD TRACKING LOG - ENVIRONMENT {env_id} ===\n")
                        f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n\n")
            
            # Get environment instance for detailed state access
            env = self.model.get_env().envs[env_id].unwrapped
            
            # Update step counter for current episode
            self.env_data[env_id]["current_episode_step"] += 1
            
            # Terminal Reaching Detection
            # ==========================
            # Check multiple sources for terminal reaching to ensure robust detection
            if info.get('reached_terminal', False) or (hasattr(env, 'reached_terminal_area') and env.reached_terminal_area):
                self.env_data[env_id]["terminal_reached"] = True
                
            # Track rooms visited during this episode
            current_room = env._get_current_room()
            self.env_data[env_id]["rooms_visited"].add(current_room)
            
            # Door Creation Event Tracking
            # ============================
            # Track when agents create doors between rooms (not terminal connections)
            if ('created_door' in info and info['created_door'] and 
                'door_connection_from' in info and 'door_connection_to' in info and
                info['door_connection_to'] != 'Terminal'):
                
                # Extract door connection information
                from_room = f"room{info['door_connection_from']}"
                to_room = f"room{info['door_connection_to']}"
                
                current_room = env._get_current_room()
                pos = [round(p, 2) for p in env.agent.pos]
                
                # Record door creation (terminal status will be updated at episode end)
                with open(self.door_csv_files[env_id], "a") as f:
                    f.write(f"{self.env_data[env_id]['current_episode']},{env.step_count},{pos[0]},{pos[1]},{pos[2]},{current_room},{from_room},{to_room},True,?\n")
                
                # Log to human-readable file
                with open(self.log_files[env_id], "a") as f:
                    f.write(f"Episode {self.env_data[env_id]['current_episode']} - Step {env.step_count}: Door created at position {pos} in {current_room} (from {from_room} to {to_room})\n")
            
            # Terminal State Event Tracking
            # =============================
            # Separately track when agents reach the terminal location
            if info.get('reached_terminal', False) or (hasattr(env, 'reached_terminal_area') and env.reached_terminal_area):
                current_room = env._get_current_room()
                pos = [round(p, 2) for p in env.agent.pos]
                reward = info.get('reward_terminal', 0)
                
                # Record terminal reaching event with reward and step information
                with open(self.terminal_csv_files[env_id], "a") as f:
                    f.write(f"{self.env_data[env_id]['current_episode']},{env.step_count},{pos[0]},{pos[1]},{pos[2]},{current_room},{reward},{self.env_data[env_id]['current_episode_step']}\n")
                
                # Log to human-readable file
                with open(self.log_files[env_id], "a") as f:
                    f.write(f"Episode {self.env_data[env_id]['current_episode']} - Step {env.step_count}: TERMINAL REACHED at position {pos} in {current_room} (reward: {reward}, total steps: {self.env_data[env_id]['current_episode_step']})\n")
            
            # Episode Boundary Detection and Data Recording
            # =============================================
            # Detect new episodes and record previous episode data
            if 'episode' in info:
                # Record outcome data for the completed episode (if not the first)
                if self.env_data[env_id]["current_episode"] > 0:
                    # Compile episode outcome statistics
                    terminal_reached = 1 if self.env_data[env_id]["terminal_reached"] else 0
                    rooms_visited = len(self.env_data[env_id]["rooms_visited"])
                    total_steps = self.env_data[env_id]["current_episode_step"]
                    
                    # Save episode outcome to CSV
                    with open(self.episode_outcomes_files[env_id], "a") as f:
                        f.write(f"{self.env_data[env_id]['current_episode']},{terminal_reached},{total_steps},{rooms_visited}\n")
                    
                    # Update door creation records with terminal reaching status
                    self._update_door_creation_terminal_status(env_id, self.env_data[env_id]["current_episode"], terminal_reached)
                
                # Reset tracking variables for new episode
                self.env_data[env_id]["episode_count"] += 1
                self.env_data[env_id]["current_episode"] = self.env_data[env_id]["episode_count"]
                self.env_data[env_id]["terminal_reached"] = False
                self.env_data[env_id]["rooms_visited"] = set()
                self.env_data[env_id]["current_episode_step"] = 0
            
            # Episode Termination Logging
            # ===========================
            # Log episode boundaries and outcomes
            if info.get('reached_terminal', False) or info.get('max_steps_reached', False):
                with open(self.log_files[env_id], "a") as f:
                    terminal_status = "TERMINAL REACHED" if self.env_data[env_id]["terminal_reached"] else "TERMINAL NOT REACHED"
                    f.write(f"\nEnd of Episode {self.env_data[env_id]['current_episode']} - {terminal_status} - Total Steps: {self.env_data[env_id]['current_episode_step']}\n\n")
        
        return True
    
    def _update_door_creation_terminal_status(self, env_id, episode, terminal_reached):
        """
        Update door creation records with terminal reaching status after episode completion.
        This correlates door creation events with episode success.
        
        Args:
            env_id (int): Environment identifier
            episode (int): Episode number to update
            terminal_reached (int): 1 if terminal was reached, 0 otherwise
        """
        # Read existing door creation data
        door_csv_path = self.door_csv_files[env_id]
        with open(door_csv_path, 'r') as f:
            lines = f.readlines()
        
        # Update terminal status for the specified episode
        updated_lines = []
        for i, line in enumerate(lines):
            if i == 0:  # Preserve header line
                updated_lines.append(line)
                continue
                
            parts = line.strip().split(',')
            if len(parts) >= 10 and int(parts[0]) == episode and parts[9] == '?':
                # Update the terminal_reached field (last column)
                parts[9] = str(terminal_reached)
                updated_lines.append(','.join(parts) + '\n')
            else:
                updated_lines.append(line)
        
        # Write updated data back to file
        with open(door_csv_path, 'w') as f:
            f.writelines(updated_lines)

class TrackerCallback(BaseCallback):
    """
    Detailed callback for tracking door creation positions and reward components.
    Creates comprehensive logs with reward breakdowns and position tracking.
    Useful for understanding agent behavior patterns and reward optimization.
    """
    def __init__(self, log_dir="./logs", verbose=0):
        super(TrackerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Track data for each environment separately
        self.env_data = {}       # Episode data storage per environment
        self.log_files = {}      # Log file paths per environment
        self.timestamp = int(time.time())  # Unique timestamp for file naming
    
    def _on_step(self) -> bool:
        """
        Process each step to track door creation, rewards, and agent positions.
        Creates detailed logs for post-training analysis.
        
        Returns:
            bool: True to continue training
        """
        # Process all environments in vectorized setup
        for i, info in enumerate(self.locals['infos']):
            env_id = i  # Environment identifier
            
            # Initialize environment tracking if new
            if env_id not in self.env_data:
                self.env_data[env_id] = {
                    "episode_count": 0,
                    "current_episode_data": {
                        "doors": [],      # Door creation events
                        "rewards": [],    # Reward component tracking
                        "positions": []   # Agent position samples
                    }
                }
                
                # Create separate log file for this environment
                self.log_files[env_id] = f"{self.log_dir}/tracking_env{env_id}_{self.timestamp}.txt"
                
                # Initialize log file with header
                with open(self.log_files[env_id], "w") as f:
                    f.write(f"=== DOOR AND REWARD TRACKING LOG - ENVIRONMENT {env_id} ===\n")
                    f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n\n")
            
            # Episode Boundary Detection
            # ==========================
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
            
            # Door Creation Event Tracking
            # ============================
            if 'created_door' in info and info['created_door']:
                # Get environment instance for detailed information
                env = self.model.get_env().envs[env_id].unwrapped
                
                # Extract door connection information
                if 'door_connection_from' in info and 'door_connection_to' in info:
                    connection_info = f"from room{info['door_connection_from']} to room{info['door_connection_to']}"
                else:
                    connection_info = self._get_latest_connection_info(env)
                
                # Check if this is a terminal door connection
                is_terminal = info.get('terminal_door_creation', False) or info.get('reached_terminal', False)
                if is_terminal:
                    connection_info = f"from {env._get_current_room()} to Terminal State"
                
                # Store door creation data
                door_data = {
                    "step": env.step_count,
                    "position": [round(pos, 2) for pos in env.agent.pos],
                    "room": env._get_current_room(),
                    "connection": connection_info,
                    "is_terminal": is_terminal
                }
                self.env_data[env_id]["current_episode_data"]["doors"].append(door_data)
            
            # Reward Component Tracking
            # ========================
            # Get environment instance to access reward scales
            env = self.model.get_env().envs[env_id].unwrapped
            
            # Extract raw reward components from step info
            raw_reward_components = {
                key: info[key] for key in [
                    'reward_orientation',     # Orientation toward target
                    'reward_distance', 'punishment_distance',  # Distance-based rewards/penalties
                    'penalty_stagnation', 'punishment_time',   # Time and movement penalties
                    'reward_hallway', 'reward_connection',     # Navigation rewards
                    'reward_terminal', 'punishment_terminal',  # Terminal-related rewards
                    'punishment_room', 'episode_rewards'       # Room penalties and total rewards
                ] if key in info
            }
            
            # Apply reward scales to get actual contribution to total reward
            scaled_reward_components = {}
            if raw_reward_components:
                # Scale each reward component according to environment configuration
                reward_scale_mappings = {
                    'reward_orientation': 'reward_orientation_scale',
                    'reward_distance': 'reward_distance_scale',
                    'punishment_distance': 'punishment_distance_scale',
                    'penalty_stagnation': 'penalty_stagnation_scale',
                    'punishment_time': 'punishment_time_scale',
                    'reward_hallway': 'reward_hallway_scale',
                    'reward_connection': 'reward_connection_scale',
                    'reward_terminal': 'reward_terminal_scale',
                    'punishment_terminal': 'punishment_terminal_scale',
                    'punishment_room': 'punishment_room_scale'
                }
                
                # Apply scaling to each component
                for component, raw_value in raw_reward_components.items():
                    if component in reward_scale_mappings:
                        scale_key = reward_scale_mappings[component]
                        scaled_reward_components[component] = raw_value * env.reward_scales[scale_key]
                    else:
                        # Include unscaled components (like episode_rewards)
                        scaled_reward_components[component] = raw_value
                
                # Store reward data for this step
                reward_data = {
                    "step": info.get('step_count', env.step_count),
                    "components": scaled_reward_components,
                    "room": info.get('current_room', env._get_current_room())
                }
                self.env_data[env_id]["current_episode_data"]["rewards"].append(reward_data)
            
            # Position Tracking
            # ================
            # Sample agent positions every 10 steps for trajectory analysis
            if hasattr(env, 'step_count') and env.step_count % 10 == 0:
                pos_data = {
                    "step": env.step_count,
                    "position": [round(pos, 2) for pos in env.agent.pos],
                    "room": env._get_current_room()
                }
                self.env_data[env_id]["current_episode_data"]["positions"].append(pos_data)
            
            # Episode Termination Handling
            # ============================
            if info.get('reached_terminal', False) or info.get('max_steps_reached', False):
                # Add terminal door creation if terminal reached but no terminal door recorded
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
                
                # Save episode data and reset for next episode
                self._save_episode_data(env_id)
                self.env_data[env_id]["current_episode_data"] = {
                    "doors": [],
                    "rewards": [],
                    "positions": []
                }
        
        return True
    
    def _get_latest_connection_info(self, env):
        """
        Extract door connection information from environment state.
        
        Args:
            env: Environment instance
            
        Returns:
            str: Description of the door connection
        """
        # Check for stored connection information
        if hasattr(env, 'last_created_connection'):
            from_room = env.last_created_connection["from"]
            to_room = env.last_created_connection["to"]
            return f"from room{from_room} to room{to_room}"
        
        # Search through environment connections
        for (x_range, z_range), data in env.connections.items():
            if data['created']:
                connection_str = data['connection']
                rooms = connection_str.split('-')
                return f"from room{rooms[0]} to room{rooms[1]}"
        
        return "unknown connection"
    
    def _save_episode_data(self, env_id):
        """
        Save comprehensive episode data to log file for analysis.
        
        Args:
            env_id (int): Environment identifier
        """
        log_file = self.log_files[env_id]
        
        with open(log_file, "a") as f:
            f.write(f"\n\n=== EPISODE {self.env_data[env_id]['episode_count']} ===\n")
            
            # Door Creation Events Summary
            # ===========================
            f.write("\n--- DOOR CREATIONS ---\n")
            if not self.env_data[env_id]["current_episode_data"]["doors"]:
                f.write("No doors created in this episode.\n")
            else:
                for door in self.env_data[env_id]["current_episode_data"]["doors"]:
                    # Distinguish between regular doors and terminal connections
                    if door.get('is_terminal', False):
                        f.write(f"Step {door['step']}: TERMINAL CONNECTION created at position {door['position']} in {door['room']} ({door['connection']})\n")
                    else:
                        f.write(f"Step {door['step']}: Door created at position {door['position']} in {door['room']} ({door['connection']})\n")
            
            # Agent Position Tracking Summary
            # ==============================
            f.write("\n--- AGENT POSITIONS (every 10 steps) ---\n")
            if not self.env_data[env_id]["current_episode_data"]["positions"]:
                f.write("No position data recorded.\n")
            else:
                # Sort positions chronologically
                sorted_positions = sorted(self.env_data[env_id]["current_episode_data"]["positions"], key=lambda x: x["step"])
                for pos in sorted_positions:
                    f.write(f"Step {pos['step']}: Position {pos['position']} in {pos['room']}\n")
            
            # Reward Component Analysis
            # ========================
            f.write("\n--- REWARD SUMMARY (SCALED VALUES) ---\n")
            if not self.env_data[env_id]["current_episode_data"]["rewards"]:
                f.write("No rewards recorded in this episode.\n")
            else:
                # Calculate total rewards for each component
                totals = {}
                for reward_entry in self.env_data[env_id]["current_episode_data"]["rewards"]:
                    for component, value in reward_entry["components"].items():
                        if component not in totals:
                            totals[component] = 0
                        totals[component] += value
                
                # Write component totals (only non-zero values)
                f.write("Total reward components for episode (scaled values):\n")
                for component, value in totals.items():
                    if abs(value) > 0.001:  # Skip negligible values
                        f.write(f"  {component}: {value:.2f}\n")
                
                # Highlight significant reward events
                f.write("\nSignificant reward events (|value| > 10):\n")
                significant_events = False
                
                # Sort rewards chronologically
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
    Simplified callback for tracking basic reward statistics and terminal reaching rates.
    Logs aggregate statistics to Weights & Biases for monitoring training progress.
    """
    def __init__(self, verbose=0):
        super(RewardSumCallback, self).__init__(verbose)
        self.rewards_sum = 0              # Cumulative reward total
        self.steps_sum = 0                # Total steps taken
        self.episode_count = 0            # Episodes completed
        self.terminal_reached_count = 0   # Episodes where terminal was reached
        
    def _on_step(self) -> bool:
        """
        Track basic training statistics and log to W&B.
        
        Returns:
            bool: True to continue training
        """
        self.steps_sum += 1
        
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                if 'episode_rewards' in info:
                    current_reward = info['episode_rewards']
                    self.rewards_sum += current_reward
                    self.episode_count += 1
                    
                    # Track terminal reaching success
                    if info.get('terminal_reached_this_episode', False):
                        self.terminal_reached_count += 1

        # Calculate and log aggregate statistics
        average_reward_per_step = self.rewards_sum / self.steps_sum if self.steps_sum > 0 else 0
        average_reward_per_episode = self.rewards_sum / self.episode_count if self.episode_count > 0 else 0
        terminal_reached_percentage = (self.terminal_reached_count / self.episode_count * 100) if self.episode_count > 0 else 0

        # Log metrics to Weights & Biases
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
    Advanced callback for detailed episode tracking and iteration-based analysis.
    Provides comprehensive statistics organized by training iterations.
    """
    def __init__(self, log_dir, timesteps_per_iteration, verbose=0):
        super(EnhancedRewardCallback, self).__init__(verbose)
        
        # Basic tracking variables
        self.rewards_sum = 0
        self.steps_sum = 0  
        self.episode_count = 0
        self.terminal_reached_count = 0
        
        # File logging setup
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time())
        self.log_file = f"{log_dir}/episodes_per_iteration_{timestamp}.txt"
        self.timesteps_per_iteration = timesteps_per_iteration
        
        # Episode organization by training iteration
        self.iteration_data = {}  # {iteration: [episode_count, terminal_count, reward_sum]}
        self.episode_buffer = []  # Buffer to collect episodes before iteration assignment
        
        # Initialize log file with CSV headers
        with open(self.log_file, "w") as f:
            f.write("Iteration,Episodes,TerminalReached,AverageReward\n")
        
    def _on_step(self) -> bool:
        """
        Track episodes and organize them by training iterations.
        
        Returns:
            bool: True to continue training
        """
        self.steps_sum += 1
        
        if 'infos' in self.locals:
            infos = self.locals['infos']
            for info in infos:
                # Detect episode completion
                if 'episode' in info:
                    # Extract reward from Stable Baselines3 format
                    current_reward = info['episode']['r']
                    self.rewards_sum += current_reward
                    self.episode_count += 1
                    
                    # Check for terminal reaching
                    terminal_reached = info.get('reached_terminal', False) or info.get('terminal_reached_this_episode', False)
                    if terminal_reached:
                        self.terminal_reached_count += 1
                    
                    # Store episode data for later iteration assignment
                    self.episode_buffer.append({
                        'timestep': self.num_timesteps,
                        'terminal_reached': terminal_reached,
                        'reward': current_reward
                    })

        # Calculate and log current statistics
        average_reward_per_step = self.rewards_sum / self.steps_sum if self.steps_sum > 0 else 0
        average_reward_per_episode = self.rewards_sum / self.episode_count if self.episode_count > 0 else 0
        terminal_reached_percentage = (self.terminal_reached_count / self.episode_count * 100) if self.episode_count > 0 else 0

        # Log metrics to Weights & Biases with proper timestep tracking
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
        Assign completed episodes to training iterations based on timestep thresholds.
        This provides iteration-level analysis of training progress.
        """
        # Sort episodes chronologically for proper iteration assignment
        self.episode_buffer.sort(key=lambda x: x['timestep'])
        
        # Clear previous iteration data
        self.iteration_data = {}
        
        # Assign each episode to its corresponding iteration
        for episode in self.episode_buffer:
            iteration = episode['timestep'] // self.timesteps_per_iteration
            
            # Initialize iteration tracking if needed
            if iteration not in self.iteration_data:
                self.iteration_data[iteration] = [0, 0, 0.0]  # [episode_count, terminal_count, reward_sum]
            
            # Update iteration statistics
            self.iteration_data[iteration][0] += 1  # episode count
            if episode['terminal_reached']:
                self.iteration_data[iteration][1] += 1  # terminal count
            self.iteration_data[iteration][2] += episode['reward']  # reward sum
    
    def on_training_end(self):
        """
        Generate final training report with iteration-based statistics.
        Called when training completes to summarize performance.
        """
        # Process all episodes into iteration buckets
        self._assign_episodes_to_iterations()
        
        # Write per-iteration statistics to file
        for iteration in sorted(self.iteration_data.keys()):
            episodes, terminals, reward_sum = self.iteration_data[iteration]
            avg_reward = reward_sum / episodes if episodes > 0 else 0.0
            
            with open(self.log_file, "a") as f:
                f.write(f"{iteration},{episodes},{terminals},{avg_reward:.4f}\n")
        
        # Generate summary statistics
        with open(self.log_file, "a") as f:
            f.write("\n--- SUMMARY ---\n")
            
            # Calculate aggregate totals
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
        
        # Log final summary to Weights & Biases
        if total_episodes > 0:
            wandb.run.summary["final_episodes"] = total_episodes
            wandb.run.summary["final_terminals"] = total_terminals
            wandb.run.summary["final_terminal_percentage"] = (total_terminals / total_episodes * 100)
            wandb.run.summary["final_average_reward"] = (total_reward / total_episodes)

# Custom Environment Implementation
# ================================
# Main environment class that implements the escape room navigation task

class CustomEnv(MiniWorldEnv):
    """
    Custom escape room environment where an agent learns to navigate through rooms,
    create doors between them, and reach a terminal location for maximum reward.
    
    This environment extends MiniWorld to provide:
    - Multi-room layout with hallway connection system
    - Door creation mechanics through wall interaction
    - Complex reward system with multiple components
    - Observation space including lidar, orientation, and room information
    """
    
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        # World Configuration
        # ==================
        # Set world dimensions matching the multi-agent environment
        self.world_width = 18.4   # Total world width
        self.world_depth = 6.7    # Total world depth
        
        # Reward scaling configuration for different behavioral components
        self.reward_scales = reward_scales or {
            'reward_orientation_scale': 1.0,      # Facing toward target
            'reward_distance_scale': 0.0,         # Getting closer to target
            'punishment_distance_scale': 0.0,     # Moving away from target
            'penalty_stagnation_scale': 1.0,      # Staying in same position
            'punishment_time_scale': 0.0,         # Time-based penalties
            'reward_hallway_scale': 1.0,          # Being in hallway
            'reward_connection_scale': 0.0,       # Creating door connections
            'reward_terminal_scale': 1.0,         # Reaching terminal
            'punishment_terminal_scale': 0.0,     # Terminal-related penalties
            'punishment_room_scale': 0.0,         # Room-based penalties
            'wall_collision_scale': 1.0           # Wall collision penalties
        }
        
        # Environment Parameters
        # =====================
        self.terminal_location = [18.4, 5.95]  # Terminal position (right side)
        self.max_episode_steps = max_episode_steps
        self.steps_until_hallway = steps_until_hallway  # Time limit to reach hallway
        
        # Episode State Tracking
        # =====================
        self.has_reached_hallway = False   # Whether agent reached hallway this episode
        self.hallway_reward_given = False  # Whether hallway reward was already given
        self.started_in_hallway = False    # Whether agent started in hallway
        self.non_hallway_steps = 0         # Steps spent outside hallway
        self.placement_incrementer = 0     # Counter for room placement cycling
        
        # Room configurations for random agent placement
        self.room_configs = [
            # Room A (leftmost)
            {'min_x': 0.4, 'max_x': 5.6, 'min_z': 0.4, 'max_z': 4.6},
            # Room B (middle)
            {'min_x': 6.6, 'max_x': 11.8, 'min_z': 0.4, 'max_z': 4.6},
            # Room C (rightmost)
            {'min_x': 12.8, 'max_x': 18.0, 'min_z': 0.4, 'max_z': 4.6},
        ]

        # Movement and behavior tracking
        self.stagnant_steps = 0  # Counter for steps without significant movement

        # Initialize parent MiniWorld environment
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        # Environment components
        self.rooms = []           # List of room objects
        self.boxes = []           # List of obstacle boxes
        self.previous_room = None # Track room transitions
        
        # Observation Space Definition
        # ============================
        # Define what information the agent receives each step
        self.observation_space = spaces.Box(
            low=np.array([
                -1, -1,                 # Agent's direction vector (normalized)
                -1,                     # Direction alignment with target (dot product)
                0, 0, 0, 0, 0,          # Lidar measurements (5 directions)
                0,                      # Normalized step count
                0,                      # Normalized stagnation counter
                0,                      # Room category identifier
            ], dtype=np.float32),
            high=np.array([
                1, 1,                   # Agent's direction vector (normalized)
                1,                      # Direction alignment with target (dot product)
                1, 1, 1, 1, 1,          # Lidar measurements (5 directions)
                1,                      # Normalized step count
                1,                      # Normalized stagnation counter
                4,                      # Room category identifier
            ], dtype=np.float32)
        )

        # Action Space Definition
        # ======================
        # Define available actions: turn right, turn left, move forward
        self.action_space = spaces.Discrete(3)

    def _gen_world(self):
        """
        Generate the world layout with rooms, hallway, and connections.
        Called once during environment initialization.
        """
        # Agent Configuration
        self.agent.radius = 0.25  # Set agent size for collision detection

        # Room Creation
        # ============
        # Create hallway (room D) spanning the full width at the bottom
        self.roomD = self.add_rect_room(min_x=0, max_x=self.world_width, 
                                         min_z=5.2, max_z=self.world_depth)
        
        # Create three upper rooms (A, B, C) separated by walls
        self.roomA = self.add_rect_room(min_x=0, max_x=6.0, 
                                         min_z=0, max_z=5)
        
        self.roomB = self.add_rect_room(min_x=6.2, max_x=12.2, 
                                         min_z=0, max_z=5)
        
        self.roomC = self.add_rect_room(min_x=12.4, max_x=18.4, 
                                         min_z=0, max_z=5)
        
        # Store all rooms for easy access
        self.rooms.extend([self.roomA, self.roomB, self.roomC, self.roomD])

        # Agent Placement
        # ==============
        # Place agent at the predetermined starting position
        self.place_entity(self.agent, pos=self.agent_start_pos, dir=self.agent_start_dir)

        # Connection Setup
        # ===============
        # Define possible room connections for door creation
        self.room_pairs = [
            (self.roomA, self.roomD, 'A', 'D'),  # Room A to Hallway
            (self.roomB, self.roomD, 'B', 'D'),  # Room B to Hallway
            (self.roomC, self.roomD, 'C', 'D')   # Room C to Hallway
        ]

        # Optional: Add box obstacles (currently disabled)
        # self._add_boxes()

        # Generate connection points (doors will be created dynamically by agent)
        self.connections = self.generate_connections(self.room_pairs, [])
        self._gen_static_data()

    def _add_boxes(self):
        """
        Add boxes as obstacles in each room (optional feature).
        Currently disabled but can be enabled for increased complexity.
        """
        self.boxes = []
        
        # Define box positions for each room
        box_configs = {
            'A': [[2.0, 0, 1.0], [2.0, 0, 2.0]],
            'B': [[6.2, 0, 1.0], [6.2, 0, 2.0]],
            'C': [[10.4, 0, 1.0], [10.4, 0, 2.0]]
        }
        
        # Create boxes in each room
        for room_id, positions in box_configs.items():
            room = getattr(self, f'room{room_id}')
            for pos in positions:
                box = Box(color='grey', size=[0.6, 0.7, 0.6])
                self.boxes.append(box)
                self.place_entity(box, pos=pos, dir=0, room=room)

    def generate_connections(self, room_pairs, special_configs):
        """
        Generate possible connection points between adjacent rooms.
        These define where doors can be created when the agent touches walls.
        
        Args:
            room_pairs (list): List of (room1, room2, id1, id2) tuples
            special_configs (list/dict): Special configuration for connections
            
        Returns:
            dict: Dictionary mapping connection coordinates to connection data
        """
        connections = {}
        
        # Convert special_configs to dictionary format if needed
        if isinstance(special_configs, list):
            special_configs = {}
        
        def add_connection(room1, room2, id1, id2, is_vertical=False):
            """Add bidirectional connection between two rooms"""
            # Check for special configuration overrides
            special_config = special_configs.get((id1, id2)) or special_configs.get((id2, id1))
            
            if is_vertical:
                # Vertical adjacency (one room above another)
                x_start = round(max(room1.min_x, room2.min_x) + 0.5, 1)
                x_end = round(min(room1.max_x, room2.max_x) - 0.5, 1)
                
                if x_end > x_start:
                    z_top = special_config['z_range'][1] if special_config and 'z_range' in special_config else room1.max_z
                    z_bottom = special_config['z_range'][0] if special_config and 'z_range' in special_config else room2.min_z
                    
                    # Add connections for both directions
                    connections[((x_start, x_end), 
                            (round(z_top - 0.1, 1), z_top))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((x_start, x_end),
                            (z_bottom, round(z_bottom + 0.1, 1)))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
            else:
                # Horizontal adjacency (side by side)
                z_start = round(max(room1.min_z, room2.min_z) + 0.5, 1)
                z_end = round(min(room1.max_z, room2.max_z) - 0.5, 1)
                
                # Apply special z-range if configured
                if special_config and 'z_range' in special_config:
                    z_start, z_end = special_config['z_range']
                
                if z_end > z_start:
                    # Add connections for both directions
                    connections[((round(room1.max_x - 0.1, 1), room1.max_x),
                            (z_start, z_end))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((room2.min_x, round(room2.min_x + 0.1, 1)),
                            (z_start, z_end))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
        
        # Process each room pair to generate connections
        for room1, room2, id1, id2 in room_pairs:
            # Check for horizontal adjacency
            if abs(room1.max_x - room2.min_x) < 0.3:
                add_connection(room1, room2, id1, id2)
            elif abs(room2.max_x - room1.min_x) < 0.3:
                add_connection(room2, room1, id2, id1)
                
            # Check for vertical adjacency
            if abs(room1.max_z - room2.min_z) < 0.3:
                add_connection(room1, room2, id1, id2, True)
            elif abs(room2.max_z - room1.min_z) < 0.3:
                add_connection(room2, room1, id2, id1, True)
                    
        return connections
    
    def _get_current_room(self):
        """
        Determine which room the agent is currently in based on position.
        
        Returns:
            str: Room identifier ('roomA', 'roomB', 'roomC', 'roomD', or 'unknown')
        """
        agent_pos = self.agent.pos
        boundary_tolerance = 0.2  # Tolerance for room boundary detection
        
        # Check each room to see if agent is inside
        for room in self.rooms:
            if ((room.min_x - boundary_tolerance <= agent_pos[0] <= room.max_x + boundary_tolerance) and 
                (room.min_z - boundary_tolerance <= agent_pos[2] <= room.max_z + boundary_tolerance)):
                
                # Find the room's attribute name
                for attr_name, attr_value in vars(self).items():
                    if attr_name.startswith('room') and attr_value == room:
                        return attr_name
        
        # Fallback to previous room if current position is ambiguous
        if hasattr(self, 'previous_room'):
            return self.previous_room
                
        return "unknown"
    
    def _get_room_category(self, room_name):
        """
        Convert room name to numeric category for observation space.
        
        Args:
            room_name (str): Room identifier
            
        Returns:
            int: Room category (0=Hallway, 1-3=Rooms A-C, 4=Unknown)
        """
        room_mapping = {
            'roomD': 0,  # Hallway
            'roomA': 1,  # Room A
            'roomB': 2,  # Room B
            'roomC': 3   # Room C
        }
        return room_mapping.get(room_name, 4)  # 4 for unknown

    # Sensor and Perception Methods
    # ============================

    def get_wall_distance(self, pos, direction, max_distance=10):
        """
        Cast a ray from position in given direction to find distance to nearest wall.
        Used for lidar-like sensor measurements.
        
        Args:
            pos (array): Starting position
            direction (array): Direction vector
            max_distance (float): Maximum sensing distance
            
        Returns:
            float: Distance to nearest wall
        """
        step_size = 0.1
        current_pos = np.array(pos)
        
        # Cast ray step by step until hitting a wall
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
        Get distance measurements in multiple directions around the agent.
        Simulates lidar sensor for obstacle detection and navigation.
        
        Returns:
            tuple: Distances in 5 directions (right, left, forward, forward-right, forward-left)
        """
        # Get agent's current direction
        forward_dir = self.agent.dir_vec
        
        # Calculate perpendicular directions
        right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])      # 90° clockwise
        left_dir = np.array([forward_dir[2], 0, -forward_dir[0]])       # 90° counterclockwise
        
        # Calculate diagonal directions
        forward_right_dir = forward_dir + right_dir
        forward_right_dir = forward_right_dir / np.linalg.norm(forward_right_dir)
        
        forward_left_dir = forward_dir + left_dir
        forward_left_dir = forward_left_dir / np.linalg.norm(forward_left_dir)
        
        # Measure distances in all directions
        measurements = (
            self.get_wall_distance(self.agent.pos, right_dir),
            self.get_wall_distance(self.agent.pos, left_dir),
            self.get_wall_distance(self.agent.pos, forward_dir),
            self.get_wall_distance(self.agent.pos, forward_right_dir),
            self.get_wall_distance(self.agent.pos, forward_left_dir)
        )
        
        return measurements
    
    def _normalize_terminal_distance(self, distance):
        """
        Normalize distance to terminal with higher values meaning closer.
        
        Args:
            distance (float): Raw distance to terminal
            
        Returns:
            float: Normalized distance (0=far, 1=close)
        """
        max_dist = np.sqrt(self.world_width**2 + self.world_depth**2)
        return (max_dist - distance) / max_dist

    # Door Creation and Wall Interaction
    # ==================================

    def _agent_touches_wall(self, agent_nose_pos):
        """
        Check if agent is touching a wall at a valid connection point.
        
        Args:
            agent_nose_pos (array): Position of agent's front edge
            
        Returns:
            tuple or None: Connection coordinates if touching, None otherwise
        """
        touching_threshold = 0.1
        
        # Check all possible connection points
        for (x_range, z_range), data in self.connections.items():
            if (x_range[0]-touching_threshold <= agent_nose_pos[0] <= x_range[1]+touching_threshold and 
                z_range[0]-touching_threshold <= agent_nose_pos[2] <= z_range[1]+touching_threshold):
                return (x_range, z_range)
        
        return None
    
    def _create_doors(self, new_connection):
        """
        Create a physical door at the specified connection point.
        
        Args:
            new_connection (tuple): Connection coordinates where door should be created
        """
        if not hasattr(self, 'special_connection_groups'):
            self.special_connection_groups = {}

        x_range, z_range = new_connection
        connection = self.connections[new_connection]['connection']
        room1_id, room2_id = connection.split('-')
        
        # Get room objects
        room1_attr = f"room{room1_id}"
        room2_attr = f"room{room2_id}"
        room1 = getattr(self, room1_attr)
        room2 = getattr(self, room2_attr)
        
        # Door creation parameters
        door_width = 1.0
        min_offset = 0.05
        
        # Create door based on room adjacency type
        if abs(room1.max_x - room2.min_x) < 0.3 or abs(room2.max_x - room1.min_x) < 0.3:
            # Horizontal adjacency (rooms side by side)
            left_room = room1 if room1.max_x < room2.min_x else room2
            right_room = room2 if left_room == room1 else room1
            
            # Position door centered on agent's z-coordinate
            min_z = self.agent.pos[2] - door_width/2
            min_z = np.clip(min_z,
                        max(left_room.min_z, right_room.min_z) + min_offset,
                        min(left_room.max_z, right_room.max_z) - door_width - min_offset)
            
            self.connect_rooms(left_room, right_room, min_z=min_z, max_z=min_z + door_width)
        else:
            # Vertical adjacency (one room above another)
            top_room = room1 if room1.max_z > room2.max_z else room2
            bottom_room = room2 if top_room == room1 else room1
            
            # Position door centered on agent's x-coordinate
            min_x = self.agent.pos[0] - door_width/2
            min_x = np.clip(min_x,
                        max(top_room.min_x, bottom_room.min_x) + min_offset,
                        min(top_room.max_x, bottom_room.max_x) - door_width - min_offset)
            
            self.connect_rooms(bottom_room, top_room, min_x=min_x, max_x=min_x + door_width)
        
        # Update connection tracking
        current_connection = f"{room1_id}-{room2_id}"
        reverse_connection = f"{room2_id}-{room1_id}"
        
        # Store connection information for tracking
        self.last_created_connection = {
            "from": room1_id,
            "to": room2_id,
            "connection_str": current_connection
        }
        
        # Handle special connection groups (if configured)
        special_group = None
        if hasattr(self, 'special_connection_groups'):
            for group_key, connections in self.special_connection_groups.items():
                if current_connection in connections or reverse_connection in connections:
                    special_group = group_key
                    break
        
        # Mark connections as created
        if special_group:
            # Mark all connections in special group as created
            group_connections = self.special_connection_groups[special_group]
            for key, value in self.connections.items():
                if value['connection'] in group_connections:
                    self.connections[key]['created'] = True
        else:
            # Mark only this connection and its reverse as created
            for key, value in self.connections.items():
                if value['connection'] in [current_connection, reverse_connection]:
                    self.connections[key]['created'] = True

    # Environment Reset and Episode Management
    # =======================================

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode with randomized agent placement.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional reset options
            
        Returns:
            tuple: (observation_array, info_dict)
        """
        # Reset episode tracking variables
        self.has_reached_hallway = False
        self.hallway_reward_given = False
        self.stagnant_steps = 0
        self.non_hallway_steps = 0 
        self.reached_terminal_area = False
        
        # Seed Configuration
        # =================
        if seed is not None:
            self.seed(seed)
        else:
            # Generate random seed if none provided
            new_seed = int(time.time() * 1000) % 100000
            self.seed(new_seed)
    
        # Agent Placement Strategy
        # =======================
        # Cycle through rooms for systematic training
        room_index = (self.placement_incrementer // 3) % len(self.room_configs)
        room_config = self.room_configs[room_index]
        
        # Generate random position and orientation within selected room
        random_x = random.uniform(room_config['min_x'], room_config['max_x'])
        random_z = random.uniform(room_config['min_z'], room_config['max_z'])
        random_dir = random.uniform(-np.pi, np.pi)
        
        # Set agent starting state
        self.agent_start_pos = [random_x, 0, random_z]
        self.agent_start_dir = random_dir
        
        # Increment placement counter for next episode
        self.placement_incrementer += 1
        
        # Environment Reset
        # ================
        observation = super().reset(seed=self.seed_value)

        # Reset door connections (no doors exist at start)
        for connection in self.connections:
            self.connections[connection]['created'] = False

        # Check starting room and set initial flags
        current_room = self._get_current_room()
        self.started_in_hallway = current_room == 'roomD'
        self.previous_room = current_room

        # Handle hallway start case
        if self.started_in_hallway:
            self.has_reached_hallway = True
            self.hallway_reward_given = True  # Don't give hallway reward if starting there

        # Initialize distance tracking for reward calculations
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        self.previous_distance_to_terminal = distance_to_terminal

        # Return initial observation
        observation_array = self._get_observation_array()
        return observation_array, {}

    def seed(self, seed=None):
        """
        Seed the environment's random number generators.
        
        Args:
            seed (int, optional): Random seed value
            
        Returns:
            list: List containing the seed value
        """
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_observation_array(self):
        """
        Create the observation array that the agent receives each step.
        Contains normalized sensor readings, position info, and environmental state.
        
        Returns:
            np.array: Formatted observation for the agent
        """
        # Direction and Orientation Calculations
        # =====================================
        
        # Calculate vector from agent to terminal
        dx = self.terminal_location[0] - self.agent.pos[0]
        dz = self.terminal_location[1] - self.agent.pos[2]
        
        # Create normalized direction vector (with flipped z for correct orientation)
        direction_vector = np.array([dx, -dz])
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        else:
            direction_vector = np.array([1.0, 0.0])  # Default if at same position
        
        # Agent's forward direction vector (always normalized)
        agent_dir_vec = np.array([
            np.cos(self.agent.dir),
            np.sin(self.agent.dir)
        ])
        
        # Calculate alignment between agent direction and target direction
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Determine sign of angle for directional turning
        cross_z = agent_dir_vec[0] * direction_vector[1] - agent_dir_vec[1] * direction_vector[0]
        if cross_z < 0:
            angle_difference = -angle_difference
        
        # Store for reward calculations
        self.last_angle_difference = angle_difference
        self.last_dot_product = dot_product

        # Sensor Measurements
        # ==================
        
        # Get lidar distance measurements
        right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()

        # Normalize lidar distances to [0,1] range
        max_lidar_dist = 30.0
        norm_right = right_dist / max_lidar_dist
        norm_left = left_dist / max_lidar_dist
        norm_forward = forward_dist / max_lidar_dist
        norm_forward_right = forward_right_dist / max_lidar_dist
        norm_forward_left = forward_left_dist / max_lidar_dist

        # Room and Environment State
        # =========================
        
        # Get current room information
        current_room = self._get_current_room()
        room_category = self._get_room_category(current_room)
        
        # Construct final observation array
        return np.array([
            agent_dir_vec[0], agent_dir_vec[1],         # Agent direction vector components (2)
            dot_product,                                # Alignment with target (-1 to 1) (1)
            norm_right, norm_left, norm_forward,        # Lidar measurements (3)
            norm_forward_right, norm_forward_left,      # Diagonal lidar measurements (2)
            self.step_count / self.max_episode_steps,   # Normalized step count (1)
            self.stagnant_steps / 100,                  # Normalized stagnation counter (1)
            room_category,                              # Room category identifier (1)
        ], dtype=np.float32)

    def step(self, action):
        """
        Execute one environment step with the given action.
        Applies complex reward function and checks for episode termination.
        
        Args:
            action (int): Action to execute (0=turn right, 1=turn left, 2=move forward)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Store previous state for movement detection
        previous_agent_pos = np.array(self.agent.pos)
    
        # Execute action in parent environment
        observation, reward, terminated, truncated, info = super().step(action)

        # Initialize reward components
        reward_components = {
            'reward_orientation': 0,
            'reward_distance_terminal': 0,
            'punishment_distance_terminal': 0,
            'punishment_time': 0,
            'reward_hallway': 0,
            'reward_connection': 0,
            'reward_terminal': 0,
            'punishment_terminal': 0,
            'punishment_room': 0,
            'penalty_stagnation': 0,
            'wall_collision_penalty': 0
        }

        # Agent State Analysis
        # ===================
        
        # Get agent's nose position for wall interaction detection
        agent_nose_pos = self.agent.pos + self.agent.dir_vec * self.agent.radius
        current_agent_pos = np.array(self.agent.pos)
        
        # Calculate distance to terminal for reward computation
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )

        # Direction and Orientation Analysis
        # =================================
        
        # Vector from agent to terminal (corrected orientation)
        direction_vector = np.array([
            self.terminal_location[0] - self.agent.pos[0],
            -(self.terminal_location[1] - self.agent.pos[2])  # Flipped z-component for correct orientation
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
        
        # Calculate dot product for orientation reward
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Movement and Stagnation Detection
        # ================================
        
        # Check if agent actually moved
        position_changed = np.linalg.norm(current_agent_pos - previous_agent_pos) > 0.1

        # Update stagnation tracking
        if position_changed:
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        # Apply stagnation penalty if agent is stuck too long
        if self.stagnant_steps >= 100:
            reward_components['penalty_stagnation'] = -100
            info['penalty_stagnation'] = reward_components['penalty_stagnation']
            info['stagnation_penalty_applied'] = True
            self.stagnant_steps = 0  # Reset counter after penalty

        # Reward Component Calculations
        # =============================
        
        # 1. Orientation Reward
        # Penalize agent for not facing toward terminal
        if dot_product < np.cos(np.pi / 9):  # Within 20 degrees
            reward_components['reward_orientation'] = -0.1
        
        # 2. Time Penalty
        # Small constant penalty to encourage faster completion
        reward_components['punishment_time'] = -0.5

        # 3. Distance-based Rewards (currently disabled via reward scales)
        if hasattr(self, 'previous_distance_to_terminal') and position_changed:
            if distance_to_terminal < self.previous_distance_to_terminal:
                if dot_product < np.cos(np.pi / 6):  # Within 30 degrees
                    reward_components['reward_distance_terminal'] = 0.5
                else:
                    reward_components['reward_distance_terminal'] = -0.1

            if distance_to_terminal > self.previous_distance_to_terminal:
                reward_components['punishment_distance_terminal'] = -0.5

        # Update distance tracking
        if position_changed:
            self.previous_distance_to_terminal = distance_to_terminal

        # Room-based Rewards and Penalties
        # ===============================
        
        current_room = self._get_current_room()
        is_in_hallway = current_room == 'roomD'
        
        if is_in_hallway:
            # Small continuous reward for being in hallway
            reward_components['reward_hallway'] = 0.05
            self.non_hallway_steps = 0
            
            # One-time large reward for first hallway visit
            first_hallway_visit = not self.has_reached_hallway
            
            if (first_hallway_visit and 
                not self.started_in_hallway and 
                not self.hallway_reward_given):
                # Large one-time reward can be enabled here
                # reward_components['reward_hallway'] += 100
                self.hallway_reward_given = True
            
            self.has_reached_hallway = True
        else:
            # Track time spent outside hallway
            self.non_hallway_steps += 1
            
            # Apply penalty for spending too long outside hallway
            if self.non_hallway_steps >= 100:
                # Room penalty can be enabled via reward scales
                # reward_components['punishment_room'] = -100
                self.non_hallway_steps = 0

        # Hallway Timeout Check
        # ====================
        
        # Truncate episode if agent doesn't reach hallway in time
        if not self.has_reached_hallway and self.step_count >= self.steps_until_hallway:
            truncated = True
            reward_components['punishment_terminal'] = -500
            info['punishment_terminal'] = reward_components['punishment_terminal']

        # Door Creation Mechanics
        # ======================
        
        # Check if agent is touching a wall at a connection point
        new_connection = self._agent_touches_wall(agent_nose_pos)
        if new_connection is not None and not self.connections[new_connection]['created']:
            # Create door and update environment
            self._create_doors(new_connection)
            # reward_components['reward_connection'] = 10  # Can be enabled via reward scales
            self._gen_static_data()
            self._render_static()  # Force visual update
            
            # Add door creation info
            info['created_door'] = True
            connection_info = self.connections[new_connection]['connection'].split('-')
            info['door_connection_from'] = connection_info[0]
            info['door_connection_to'] = connection_info[1]

        # Wall Collision Detection
        # =======================
        
        collision_threshold = 0.075
        is_touching_connection = new_connection is not None

        # Only penalize wall collisions when not creating doors
        if not is_touching_connection:
            # Check all lidar measurements for wall proximity
            right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()
            
            min_distance = min(forward_dist, right_dist, left_dist, forward_right_dist, forward_left_dist)
            
            if min_distance < collision_threshold:
                # Calculate penalty based on proximity to wall
                reward_components['wall_collision_penalty'] = -1 * (1 - (min_distance / collision_threshold))
                info['wall_collision'] = True
                info['wall_collision_penalty'] = reward_components['wall_collision_penalty']

        # Hallway Timeout Handling
        # ========================
        
        if not self.has_reached_hallway and self.step_count >= self.steps_until_hallway:
            truncated = True
            info['truncation_reason'] = 'hallway_timeout'
            reward_components['punishment_terminal'] = -100

        # Terminal State Detection
        # =======================
        
        # Check if agent has reached the terminal location
        terminal_x, terminal_z = self.terminal_location
        if (terminal_x - 0.2 <= agent_nose_pos[0] <= terminal_x + 0.2 and 
            terminal_z - 0.2 <= agent_nose_pos[2] <= terminal_z + 0.2):
            terminated = True
            self.reached_terminal_area = True
            
            # Terminal reward (can be step-dependent or fixed)
            reward_components['reward_terminal'] = 500
            info['reached_terminal'] = True

        # Episode Length Limit
        # ===================
        
        # Truncate if maximum steps reached
        if self.step_count >= self.max_episode_steps:
            truncated = True
            reward_components['punishment_terminal'] = -100
            info['max_steps_reached'] = True

        # Final Reward Calculation
        # =======================
        
        # Apply reward scaling to get final reward
        final_reward = (
            self.reward_scales['reward_orientation_scale'] * reward_components['reward_orientation'] +
            self.reward_scales['reward_distance_scale'] * reward_components['reward_distance_terminal'] +
            self.reward_scales['punishment_distance_scale'] * reward_components['punishment_distance_terminal'] +
            self.reward_scales['penalty_stagnation_scale'] * reward_components['penalty_stagnation'] +
            self.reward_scales['punishment_time_scale'] * reward_components['punishment_time'] +
            self.reward_scales['reward_hallway_scale'] * reward_components['reward_hallway'] +
            self.reward_scales['reward_connection_scale'] * reward_components['reward_connection'] +
            self.reward_scales['reward_terminal_scale'] * reward_components['reward_terminal'] +
            self.reward_scales['punishment_terminal_scale'] * reward_components['punishment_terminal'] +
            self.reward_scales['punishment_room_scale'] * reward_components['punishment_room'] +
            self.reward_scales['wall_collision_scale'] * reward_components['wall_collision_penalty']
        )

        # Information Logging
        # ==================
        
        # Add all reward components to info for tracking
        info.update(reward_components)
        info.update({
            'episode_rewards': final_reward,
            'step_count': self.step_count,
            'current_room': current_room
        })

        # Return step results
        observation_array = self._get_observation_array()
        return observation_array, final_reward, terminated, truncated, info

# Robust Environment Wrapper
# ==========================

class RobustNoRenderEnv(CustomEnv):
    """
    Headless version of the environment that disables rendering for server training.
    Prevents graphics-related errors on systems without display capabilities.
    """
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        # Configure minimal window settings
        kwargs.update({
            'window_width': 1,
            'window_height': 1
        })
        super().__init__(max_episode_steps=max_episode_steps, 
                        steps_until_hallway=steps_until_hallway,
                        reward_scales=reward_scales, **kwargs)
        self.render_mode = 'none'
        
    def render_obs(self, vis_fb=None):
        """Disable observation rendering for headless operation"""
        return np.zeros((3, 84, 84), dtype=np.uint8)
    
    def _render_static(self):
        """Disable static rendering for headless operation"""
        pass
        
    def _render_dynamic(self):
        """Disable dynamic rendering for headless operation"""
        pass

# Training Infrastructure
# ======================

class CheckpointCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints at regular intervals.
    Saves models every N iterations for recovery and analysis purposes.
    """
    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.iteration_count = 0

    def _on_step(self) -> bool:
        """
        Check if it's time to save a checkpoint and save if needed.
        
        Returns:
            bool: True to continue training
        """
        # Ensure wandb config is available
        if not hasattr(wandb, 'config') or 'iterations' not in wandb.config:
            return True
            
        # Calculate current iteration based on timesteps
        timesteps_per_iteration = self.locals.get('total_timesteps') // wandb.config['iterations']
        current_iteration = self.num_timesteps // timesteps_per_iteration

        # Save checkpoint every 10 iterations
        if current_iteration > self.iteration_count and current_iteration % 10 == 0:
            self.iteration_count = current_iteration
            path = f"{self.save_path}/{self.name_prefix}_iteration_{current_iteration}"
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint at iteration {current_iteration}")
        return True

# Environment Factory Function
# ===========================

def make_env(max_episode_steps, steps_until_hallway, seed, reward_scales=None):
    """
    Create a wrapped environment instance with monitoring and seeding.
    
    Args:
        max_episode_steps (int): Maximum steps per episode
        steps_until_hallway (int): Time limit to reach hallway
        seed (int): Random seed for reproducibility
        reward_scales (dict): Reward component scaling factors
        
    Returns:
        function: Environment creation function for vectorized environments
    """
    def _init():
        try:
            # Create robust environment instance
            env = RobustNoRenderEnv(
                max_episode_steps=max_episode_steps, 
                steps_until_hallway=steps_until_hallway, 
                reward_scales=reward_scales
            )
            # Wrap with monitoring for episode statistics
            env = Monitor(env)
            # Set random seed
            env.unwrapped.seed(seed)
            return env
        except Exception as e:
            print(f"Error initializing environment with seed {seed}:")
            print(traceback.format_exc())
            raise
    return _init

# Main Training Function
# =====================

def run_experiment(config):
    """
    Execute a complete training experiment with the given configuration.
    Sets up environment, model, callbacks, and runs training loop.
    
    Args:
        config (dict): Experiment configuration dictionary
    """
    # Expand configuration with training parameters
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": config['timesteps'] * config['iterations'],
        "env_name": "CustomEnv-EscapeRoom-v1",
        **config
    }
    
    # Experiment identification
    run_name = "escape-room-v2_likemulti_middlebig_50termrew"
    
    # Initialize Weights & Biases tracking
    with wandb.init(project="simple_multi-single",
                    name=run_name,
                    config=config,
                    sync_tensorboard=True) as run:
        
        # Save script for reproducibility
        wandb.save("env_sarl_v2.py")
        
        # Environment Setup
        # ================
        
        # Create vectorized environments with unique seeds
        num_envs = 1
        base_seed = config['seed']
        
        # Generate unique seeds for each environment instance
        env_seeds = [base_seed + i * 1000 for i in range(num_envs)]
        
        # Create vectorized environment
        envs = DummyVecEnv([
            make_env(
                config['max_episode_steps'], 
                config['steps_until_hallway'], 
                env_seeds[i],
                {k: config[k] for k in config if k.endswith('_scale')}  # Extract reward scales
            ) for i in range(num_envs)
        ])
        
        # Model Creation
        # =============
        
        # Create PPO model with specified configuration
        model = PPO("MlpPolicy",
                    envs,
                    verbose=1,
                    tensorboard_log=f"runs/simple_match_{run_name}_{wandb.run.id}",
                    seed=base_seed,
                    device="auto")
        
        # Directory Setup
        # ==============
        
        # Create directories for logging and checkpoints
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        tracking_dir = f"tracking_logs/{wandb.run.id}"
        tracking_light_dir = f"tracking_doors_logs/{wandb.run.id}"
        script_dir = f"script_logs/{wandb.run.id}"
        
        for dir_path in [script_dir, checkpoint_dir, tracking_dir, tracking_light_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save script backup for reproducibility
        script_path = os.path.abspath(__file__)
        script_backup_path = os.path.join(script_dir, "train_backup.py")
        try:
            shutil.copy2(script_path, script_backup_path)
            print(f"Saved training script backup to: {script_backup_path}")
        except Exception as e:
            print(f"Failed to save script backup: {e}")

        # Callback Setup
        # =============
        
        # Create enhanced reward tracking callback
        enhanced_reward_callback = EnhancedRewardCallback(
            log_dir=f"episode_logs/{wandb.run.id}",
            timesteps_per_iteration=config['timesteps'],
            verbose=1
        )

        # Configure all training callbacks
        callbacks = [
            WandbCallback(
                model_save_path=f"models/{wandb.run.id}_PPO_{run_name}",
                verbose=2,
            ),
            enhanced_reward_callback,
            CheckpointCallback(
                save_freq=1,
                save_path=checkpoint_dir,
                name_prefix=f"checkpoint_{wandb.run.id}",
                verbose=1
            ),
            TrackerCallback(log_dir=tracking_dir, verbose=1),
            ImprovedTrackerCallback(log_dir=tracking_light_dir, verbose=1),
        ]
        
        try:
            # Training Execution
            # =================
            
            # Run training for specified number of timesteps
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
            # Cleanup
            # =======
            
            # Clean up resources
            envs.close()
            del model
            gc.collect()

# Main Execution
# =============

def main():
    """
    Main function to configure and run the training experiment.
    Sets up multiprocessing and executes training with default parameters.
    """
    # Configure multiprocessing for stable training
    try:
        multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        pass

    # Generate random seed for experiment uniqueness
    random_seed = int(time.time()) % 10000
    
    # Default experimental configuration
    config = {
        'seed': random_seed,                    # Random seed for reproducibility within run
        'max_episode_steps': 500,              # Maximum steps per episode
        'timesteps': 12000,                    # Timesteps per training iteration
        'iterations': 1000,                    # Number of training iterations
        'steps_until_hallway': 500,            # Time limit to reach hallway
        
        # Reward component scaling factors
        'reward_orientation_scale': 1.0,       # Weight for orientation rewards
        'reward_distance_scale': 0.0,          # Weight for distance rewards (disabled)
        'punishment_distance_scale': 0.0,      # Weight for distance penalties (disabled)
        'penalty_stagnation_scale': 1.0,       # Weight for stagnation penalties
        'punishment_time_scale': 0.0,          # Weight for time penalties (disabled)
        'reward_hallway_scale': 1.0,           # Weight for hallway rewards
        'reward_connection_scale': 0.0,        # Weight for door creation rewards (disabled)
        'reward_terminal_scale': 0.1,          # Weight for terminal reaching rewards
        'punishment_terminal_scale': 0.0,      # Weight for terminal penalties (disabled)
        'punishment_room_scale': 0.0,          # Weight for room penalties (disabled)
        'wall_collision_scale': 1.0            # Weight for wall collision penalties
    }
    
    # Log experiment parameters
    print(f"Running experiment with random seed: {random_seed}")
    
    # Execute training experiment
    run_experiment(config)

# Script Entry Point
# =================

if __name__ == "__main__":
    main()