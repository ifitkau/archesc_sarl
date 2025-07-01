#!/usr/bin/env python3
"""
Multi-Room Escape Environment Path Visualizer

This script provides comprehensive visualization tools for analyzing agent trajectories
in the multi-room escape environment. It loads trained reinforcement learning models
and generates various types of visualizations to understand agent behavior, path
efficiency, and door creation patterns.

Key Features:
- Multiple visualization types (transparent paths, grayscale, heat maps)
- Door creation activity heat mapping
- Success/failure path analysis
- Configurable visualization parameters
- Automatic model and environment detection
- Support for different checkpoint iterations

Visualization Types:
1. Transparent Paths: Clean overlays showing successful vs failed trajectories
2. Grayscale Academic: Publication-ready visualizations with enhanced contrast
3. Heat Map Analysis: Door creation frequency visualization
4. Success-Only Analysis: Focused on successful path patterns

Usage: 
    python path_visualizer.py [model_path] [--env-file ENV_FILE] [--width WIDTH]
    python path_visualizer.py --num-episodes 100 --successful-only


Date: July 2025
"""

import sys
import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from datetime import datetime
import importlib.util
from scipy.ndimage import gaussian_filter1d

def import_environment_module(file_path):
    """
    Import the module containing environment classes from a file path.
    
    This function dynamically imports the environment module, allowing the
    visualizer to work with different versions of the environment code.
    
    Args:
        file_path: Path to the Python file containing environment classes
        
    Returns:
        Imported module object containing environment classes
        
    Raises:
        SystemExit: If the module cannot be imported
    """
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"Successfully imported environment from {file_path}")
        return module
    except Exception as e:
        print(f"ERROR: Failed to import environment from {file_path}: {e}")
        sys.exit(1)

def find_environment_file():
    """
    Automatically find an appropriate environment file in the current directory.
    
    Searches for Python files that match expected environment naming patterns,
    prioritizing certain versions over others.
    
    Returns:
        String path to the environment file
        
    Raises:
        SystemExit: If no suitable environment file is found
    """
    # Prioritized list of potential environment files
    potential_files = [
        "env_simple_learn_flex_final_match_bigger.py",  # Try bigger version first
        "env_simple_learn_flex_final_match.py"
    ]
    
    # Also look for any file starting with "env_" and ending with ".py"
    env_files = [f for f in os.listdir('.') if f.startswith('env_') and f.endswith('.py')]
    for file in env_files:
        if file not in potential_files:
            potential_files.append(file)
    
    # Check if any of the potential files exist
    for file in potential_files:
        if os.path.exists(file):
            return file
    
    print("ERROR: Could not find an environment file.")
    print("Please specify the environment file with --env-file")
    sys.exit(1)

def find_model_path(model_path=None):
    """
    Find a trained model file from command line args or by searching directories.
    
    If no model path is provided, searches common directories for .zip model files
    and returns the most recently modified one.
    
    Args:
        model_path: Optional explicit path to model file
        
    Returns:
        String path to model file, or None if not found
    """
    # Check if model path was provided and exists
    if model_path and os.path.exists(model_path):
        return model_path
    
    # Otherwise look in common directories
    search_dirs = ["models", "checkpoints", "."]
    for directory in search_dirs:
        if os.path.exists(directory):
            model_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.endswith('.zip')
            ]
            if model_files:
                # Sort by modification time to get the latest
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return model_files[0]
    
    return None

def create_wider_visualization(paths, outcomes, world_width, world_depth, 
                             output_dir="trajectories", show_all_paths=True,
                             line_width=1.0, start_size=4, no_title=False,
                             color_successful='blue', color_failed='red',
                             width=15, iteration_num=None, max_steps=None, num_episodes=None):
    """
    Create basic path visualization with custom width control.
    
    This function creates a simple visualization showing agent trajectories
    with different colors for successful and failed paths.
    
    Args:
        paths: List of trajectory paths (each path is a list of [x, y, z] positions)
        outcomes: List of boolean outcomes (True for success, False for failure)
        world_width: Width of the environment world
        world_depth: Depth of the environment world
        output_dir: Directory to save visualization files
        show_all_paths: Whether to show all paths or only successful ones
        line_width: Width of trajectory lines in the plot
        start_size: Size of starting position markers
        no_title: Whether to omit the title from the plot
        color_successful: Color string for successful paths
        color_failed: Color string for failed paths
        width: Width of the figure in inches
        iteration_num: Iteration number of the checkpoint (for filename/title)
        max_steps: Maximum steps per episode (for title)
        num_episodes: Number of episodes run (for title)
        
    Returns:
        String path to the saved visualization file, or None if no paths
    """
    if not paths:
        print("No paths to visualize.")
        return None
    
    # Calculate appropriate height based on width to maintain aspect ratio
    height = (width * 2.2)  # Wider than true aspect ratio as requested
    
    # Create figure with specified width and transparent background
    fig, ax = plt.subplots(figsize=(width, height), facecolor='none')
    
    # Remove all spines, ticks, labels for clean visualization
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set limits to match the environment dimensions
    ax.set_xlim(0, world_width)
    ax.set_ylim(world_depth, 0)  # Reversed y-axis to match expected orientation
    
    # Draw all trajectory paths
    for path, outcome in zip(paths, outcomes):
        path = np.array(path)
        x, z = path[:, 0], path[:, 2]  # x and z coordinates (skip y)
        
        # Use specified colors based on path outcome
        color = color_successful if outcome else color_failed
        
        # Plot trajectory line with specified width
        ax.plot(x, z, color=color, linewidth=line_width, alpha=0.7)
        
        # Add a small circle at the starting position
        ax.scatter(x[0], z[0], color=color, s=start_size, alpha=0.8)
    
    # Add title if not disabled
    if not no_title:
        env_size = f"{world_width}x{world_depth}"
        path_type = "All Paths" if show_all_paths else "Successful Paths"
        
        # Include iteration and steps information if available
        iteration_info = f"_iter{iteration_num}" if iteration_num else ""
        title = f"{path_type} - Environment Size: {env_size}{iteration_info}"
        ax.set_title(title, fontsize=10, pad=10)
    
    # Save the figure with informative filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct filename with additional information
    path_type_filename = "all_paths" if show_all_paths else "successful_paths"
    
    # Add iteration number and max steps to filename if available
    iteration_str = f"_iter{iteration_num}" if iteration_num else ""
    maxsteps_str = f"_maxsteps{max_steps}" if max_steps else ""
    episodes_str = f"_ep{num_episodes}" if num_episodes else ""
    
    filename = f"{output_dir}/{path_type_filename}{iteration_str}{maxsteps_str}{episodes_str}_{timestamp}.png"
    
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)
    print(f"Saved visualization to: {filename}")
    plt.close(fig)
    
    return filename

def draw_boxes(ax, env, box_color='#666666', box_edge_color='#444444', alpha=0.7):
    """
    Draw boxes (obstacles) from the environment on the visualization.
    
    Renders the box obstacles present in the environment as rectangles
    on the matplotlib axis, providing context for agent navigation.
    
    Args:
        ax: Matplotlib axis object to draw on
        env: Environment object containing box obstacle data
        box_color: Fill color for the boxes
        box_edge_color: Edge/border color for the boxes
        alpha: Transparency level (0.0 to 1.0)
    """
    # Check if environment has boxes to draw
    if not hasattr(env, 'boxes') or not env.boxes:
        return
    
    # Draw each box as a rectangle using top-down 2D projection
    for box in env.boxes:
        # Extract 3D position and size, project to 2D (x-z plane)
        x, _, z = box.pos
        width, _, depth = box.size
        
        # Create rectangle centered at the box position
        rect = plt.Rectangle(
            (x - width/2, z - depth/2),  # Bottom-left corner
            width,                       # Width (x-dimension)
            depth,                       # Height (z-dimension)
            facecolor=box_color,
            edgecolor=box_edge_color,
            linewidth=0.5,
            alpha=alpha,
            zorder=4  # Above floor, below paths
        )
        ax.add_patch(rect)

def detect_room_boundary_crossings(env, paths):
    """
    Detect when agent paths cross room boundaries to identify door creation hotspots.
    
    This function analyzes agent trajectories to find where they cross between rooms,
    which indicates where doors are being created and used. This data is used to
    generate heat maps showing the most frequently used connection points.
    
    Args:
        env: Environment object with room definitions and boundaries
        paths: List of agent trajectory paths (each path is a list of [x, y, z] positions)
    
    Returns:
        Dictionary mapping boundary identifiers to crossing data:
        - 'boundary': boundary definition (type, coordinates, room connections)
        - 'crossings': list of (x, z) coordinates where crossings occurred
        - 'count': total number of crossings for this boundary
    """
    # Define the room boundaries by analyzing room adjacencies
    boundaries = []
    
    # Go through all room pairs to find shared boundaries
    for room1_name, room1 in vars(env).items():
        if not room1_name.startswith('room') or not hasattr(room1, 'min_x'):
            continue
            
        for room2_name, room2 in vars(env).items():
            if not room2_name.startswith('room') or not hasattr(room2, 'min_x') or room1 == room2:
                continue
            
            # Check for vertical boundary (rooms adjacent along x-axis)
            if abs(room1.max_x - room2.min_x) < 0.3:
                # Room1 is to the left of Room2
                boundary = {
                    'type': 'vertical',
                    'x': (room1.max_x + room2.min_x) / 2,  # Middle of the gap
                    'min_z': max(room1.min_z, room2.min_z),
                    'max_z': min(room1.max_z, room2.max_z),
                    'rooms': (room1_name, room2_name)
                }
                boundaries.append(boundary)
            elif abs(room2.max_x - room1.min_x) < 0.3:
                # Room2 is to the left of Room1
                boundary = {
                    'type': 'vertical',
                    'x': (room2.max_x + room1.min_x) / 2,  # Middle of the gap
                    'min_z': max(room1.min_z, room2.min_z),
                    'max_z': min(room1.max_z, room2.max_z),
                    'rooms': (room2_name, room1_name)
                }
                boundaries.append(boundary)
            
            # Check for horizontal boundary (rooms adjacent along z-axis)
            if abs(room1.max_z - room2.min_z) < 0.3:
                # Room1 is above Room2
                boundary = {
                    'type': 'horizontal',
                    'z': (room1.max_z + room2.min_z) / 2,  # Middle of the gap
                    'min_x': max(room1.min_x, room2.min_x),
                    'max_x': min(room1.max_x, room2.max_x),
                    'rooms': (room1_name, room2_name)
                }
                boundaries.append(boundary)
            elif abs(room2.max_z - room1.min_z) < 0.3:
                # Room2 is above Room1
                boundary = {
                    'type': 'horizontal',
                    'z': (room2.max_z + room1.min_z) / 2,  # Middle of the gap
                    'min_x': max(room1.min_x, room2.min_x),
                    'max_x': min(room1.max_x, room2.max_x),
                    'rooms': (room2_name, room1_name)
                }
                boundaries.append(boundary)
    
    # Initialize crossing count data structure
    boundary_crossings = {}
    
    # Initialize the crossing counts for each boundary
    for i, boundary in enumerate(boundaries):
        boundary_id = f"{boundary['rooms'][0]}-{boundary['rooms'][1]}"
        boundary_crossings[boundary_id] = {
            'boundary': boundary,
            'crossings': [],  # Store coordinates of crossing points
            'count': 0
        }
    
    # Analyze all paths to detect boundary crossings
    for path in paths:
        for i in range(1, len(path)):
            prev_point = path[i-1]
            curr_point = path[i]
            
            # For each boundary, check if the path segment crosses it
            for boundary_id, data in boundary_crossings.items():
                boundary = data['boundary']
                
                if boundary['type'] == 'vertical':
                    # Check if path segment crosses the vertical boundary
                    x1, _, z1 = prev_point
                    x2, _, z2 = curr_point
                    
                    # Check if the x-coordinates straddle the boundary
                    if (x1 < boundary['x'] < x2) or (x2 < boundary['x'] < x1):
                        # Calculate the z-coordinate at the crossing point using linear interpolation
                        if x2 != x1:  # Avoid division by zero
                            z_cross = z1 + (z2-z1)*(boundary['x']-x1)/(x2-x1)
                            
                            # Check if the crossing point is within the valid range of the boundary
                            if boundary['min_z'] <= z_cross <= boundary['max_z']:
                                boundary_crossings[boundary_id]['crossings'].append((boundary['x'], z_cross))
                                boundary_crossings[boundary_id]['count'] += 1
                
                elif boundary['type'] == 'horizontal':
                    # Check if path segment crosses the horizontal boundary
                    x1, _, z1 = prev_point
                    x2, _, z2 = curr_point
                    
                    # Check if the z-coordinates straddle the boundary
                    if (z1 < boundary['z'] < z2) or (z2 < boundary['z'] < z1):
                        # Calculate the x-coordinate at the crossing point using linear interpolation
                        if z2 != z1:  # Avoid division by zero
                            x_cross = x1 + (x2-x1)*(boundary['z']-z1)/(z2-z1)
                            
                            # Check if the crossing point is within the valid range of the boundary
                            if boundary['min_x'] <= x_cross <= boundary['max_x']:
                                boundary_crossings[boundary_id]['crossings'].append((x_cross, boundary['z']))
                                boundary_crossings[boundary_id]['count'] += 1
    
    return boundary_crossings

def create_transparent_paths_visualization(env, paths=None, outcomes=None, 
                                          output_dir="viz_layouts",
                                          no_title=False, width=15,
                                          iteration_num=None, max_steps=None, num_episodes=None,
                                          successful_paths_only_for_doors=True):
    """
    Create a clean visualization with only paths and heat areas on transparent background.
    
    This is the main transparent visualization function that creates overlay-ready
    visualizations without boxes, focusing on trajectory paths and door creation
    heat maps. Designed for layering over other visualizations or presentations.
    
    Args:
        env: Environment object with world dimensions and structure
        paths: Optional list of agent paths for visualization
        outcomes: Optional list of path outcomes (success/failure)
        output_dir: Directory to save the visualization file
        no_title: Whether to omit the title
        width: Width of the figure in inches
        iteration_num: Iteration number for filename and title
        max_steps: Maximum steps per episode for title
        num_episodes: Number of episodes visualized for title
        successful_paths_only_for_doors: Whether to use only successful paths for door detection
        
    Returns:
        String path to the saved visualization file
    """
    # Get environment dimensions
    world_width = env.world_width
    world_depth = env.world_depth
    
    # Calculate height based on width keeping proper aspect ratio
    height = width * (world_depth / world_width)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(width, height), facecolor='none')
    
    # Set coordinate limits with padding
    ax.set_xlim(-0.5, world_width + 0.5)
    ax.set_ylim(world_depth + 0.5, -0.5)  # Reversed y-axis for proper orientation
    
    # Remove all visual elements for clean transparent overlay
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)  # Make axis background transparent
    
    # 1. Calculate and visualize boundary crossings with orange heat map
    if paths:
        # Use all paths for door crossing detection and heat map generation
        boundary_crossings = detect_room_boundary_crossings(env, paths)
        
        # Visualize boundary crossings as heat map
        for boundary_id, data in boundary_crossings.items():
            boundary = data['boundary']
            crossings = data['crossings']
            
            if not crossings:
                continue
            
            # Use orange for door crossings (original heat map color)
            door_color = '#FF7F00'  # Orange color for heat map visibility
            door_width = 0.25
            
            if boundary['type'] == 'vertical':
                # Process vertical wall crossings
                wall_x = boundary['x']
                z_coords = [z for _, z in crossings]
                
                # Create histogram for heat map density calculation
                hist, bin_edges = np.histogram(
                    z_coords, bins=50, range=(boundary['min_z'], boundary['max_z'])
                )
                
                # Apply Gaussian smoothing for smooth heat map
                hist_smooth = gaussian_filter1d(hist, sigma=1.0)
                
                # Normalize intensities for visualization
                max_count = max(1, np.max(hist_smooth))
                normalized = hist_smooth / max_count
                
                # Draw heat map rectangles
                for i, intensity in enumerate(normalized):
                    if intensity > 0.05:  # Only draw if there's meaningful activity
                        z_start = bin_edges[i]
                        z_end = bin_edges[i+1]
                        
                        alpha = min(0.9, 0.2 + intensity * 0.8)  # Enhanced contrast
                        
                        rect = plt.Rectangle(
                            (wall_x - door_width/2, z_start),
                            door_width,
                            z_end - z_start,
                            facecolor=door_color,
                            edgecolor='none',
                            alpha=alpha,
                            zorder=5
                        )
                        ax.add_patch(rect)
                
            else:  # horizontal boundary
                # Process horizontal wall crossings
                wall_z = boundary['z']
                x_coords = [x for x, _ in crossings]
                
                # Create histogram for heat map density calculation
                hist, bin_edges = np.histogram(
                    x_coords, bins=50, range=(boundary['min_x'], boundary['max_x'])
                )
                
                # Apply Gaussian smoothing for smooth heat map
                hist_smooth = gaussian_filter1d(hist, sigma=1.0)
                
                # Normalize intensities for visualization
                max_count = max(1, np.max(hist_smooth))
                normalized = hist_smooth / max_count
                
                # Draw heat map rectangles
                for i, intensity in enumerate(normalized):
                    if intensity > 0.05:
                        x_start = bin_edges[i]
                        x_end = bin_edges[i+1]
                        
                        alpha = min(0.9, 0.2 + intensity * 0.8)
                        
                        rect = plt.Rectangle(
                            (x_start, wall_z - door_width/2),
                            x_end - x_start,
                            door_width,
                            facecolor=door_color,
                            edgecolor='none',
                            alpha=alpha,
                            zorder=5
                        )
                        ax.add_patch(rect)
    
    # Draw obstacle boxes for context (can be disabled by not calling this function)
    draw_boxes(ax, env, box_color='#777777', box_edge_color='#555555', alpha=0.5)
    
    # 2. Draw trajectory paths with enhanced visibility
    if paths and outcomes:
        # Draw failed paths in red (background layer)
        for path, outcome in zip(paths, outcomes):
            if not outcome:  # Failed path
                path = np.array(path)
                x, z = path[:, 0], path[:, 2]
                
                ax.plot(
                    x, z, 
                    color='red',
                    linewidth=0.6,  # Thinner for failed paths
                    alpha=0.4,      # More transparent
                    zorder=6        # Above boxes
                )

                # Mark starting position with small red dot
                ax.scatter(x[0], z[0], color='red', s=5, alpha=0.4, zorder=6)
        
        # Draw successful paths in blue (foreground layer)
        for path, outcome in zip(paths, outcomes):
            if outcome:  # Successful path
                path = np.array(path)
                x, z = path[:, 0], path[:, 2]
                
                ax.plot(
                    x, z, 
                    color='blue',
                    linewidth=2.0,  # Thicker for successful paths
                    alpha=0.85,     # More opaque for emphasis
                    zorder=7        # Above boxes and failed paths
                )

                # Mark starting position with blue dot
                ax.scatter(x[0], z[0], color='blue', s=6, alpha=0.6, zorder=7)
    
    # Mark terminal location prominently
    if hasattr(env, 'terminal_location'):
        term_x, term_z = env.terminal_location
        ax.scatter(
            term_x, term_z, 
            marker='*',
            s=120,
            color='green',
            edgecolor='darkgreen',
            linewidth=0.8,
            zorder=10,
            label='Terminal'
        )
    
    # Add informative title if not disabled
    if not no_title:
        # Count boundary crossings for title information
        if 'boundary_crossings' in locals():
            crossing_count = sum(data['count'] for data in boundary_crossings.values())
        else:
            crossing_count = 0
            
        # Create informative title parts
        iteration_str = f"Iteration {iteration_num}" if iteration_num else ""
        steps_str = f"Max Steps: {max_steps}" if max_steps else ""
        episodes_str = f"Episodes: {num_episodes}" if num_episodes else ""
        crossings_str = f"{crossing_count} crossings" if crossing_count > 0 else ""
        
        # Combine title parts
        title_parts = [p for p in [iteration_str, steps_str, episodes_str, crossings_str] if p]
        title = "Paths and Crossings" if not title_parts else "Paths and Crossings - " + " - ".join(title_parts)
        
        # Add title to visualization
        ax.set_title(
            title,
            fontsize=11,
            color='black',
            pad=10
        )
    
    # Save with transparent background maintained
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename
    iteration_part = f"_iter{iteration_num}" if iteration_num else ""
    steps_part = f"_steps{max_steps}" if max_steps else ""
    episodes_part = f"_ep{num_episodes}" if num_episodes else ""
    
    filename = f"{output_dir}/transparent_paths{iteration_part}{steps_part}{episodes_part}_{timestamp}.png"
    
    # Save with high quality and transparency maintained
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved transparent paths visualization to: {filename}")
    plt.close(fig)
    
    return filename

def create_transparent_paths_visualization_successful_only(env, paths=None, outcomes=None, 
                                          output_dir="viz_layouts",
                                          no_title=False, width=15,
                                          iteration_num=None, max_steps=None, num_episodes=None):
    """
    Create a visualization focusing on successful path analysis with comprehensive heat mapping.
    
    This specialized visualization uses only successful paths for door heat map calculation
    while still showing all paths for context. This provides cleaner heat maps that focus
    on effective door usage patterns rather than including failed attempts.
    
    Args:
        env: Environment object with world dimensions and structure
        paths: List of all agent paths (both successful and failed)
        outcomes: List of path outcomes (True for success, False for failure)
        output_dir: Directory to save the visualization file
        no_title: Whether to omit the title
        width: Width of the figure in inches
        iteration_num: Iteration number for filename and title
        max_steps: Maximum steps per episode for title
        num_episodes: Number of episodes visualized for title
        
    Returns:
        String path to the saved visualization file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from datetime import datetime
    import os
    
    # Ensure we have paths and outcomes for analysis
    if not paths or not outcomes:
        print("No paths to visualize.")
        return None
    
    # Filter to get just successful paths for boundary crossing detection
    successful_paths = [path for path, outcome in zip(paths, outcomes) if outcome]
    successful_count = len(successful_paths)
    
    print(f"Visualizing {len(paths)} paths with {successful_count} successful paths")
    print(f"Using only successful paths for door heat map calculation")
    
    # Get environment dimensions
    world_width = env.world_width
    world_depth = env.world_depth
    
    # Calculate height based on width keeping proper aspect ratio
    height = width * (world_depth / world_width)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(width, height), facecolor='none')
    
    # Set coordinate limits with padding
    ax.set_xlim(-0.5, world_width + 0.5)
    ax.set_ylim(world_depth + 0.5, -0.5)  # Reversed y-axis for proper orientation
    
    # Remove all visual elements for clean transparent overlay
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)  # Make axis background transparent
    
    # 1. Calculate boundary crossings using ONLY successful paths for clean heat maps
    boundary_crossings = {}
    crossing_count = 0

    if successful_paths:
        # Generate heat map data from successful paths only
        boundary_crossings = detect_room_boundary_crossings(env, successful_paths)
        
        # Count total crossings for title information
        for data in boundary_crossings.values():
            crossing_count += len(data['crossings'])
            
        print(f"Detected {crossing_count} boundary crossings from {successful_count} successful paths")
        
        # Visualize boundary crossings as focused heat map
        for boundary_id, data in boundary_crossings.items():
            boundary = data['boundary']
            crossings = data['crossings']
            
            if not crossings:
                continue
            
            # Use orange for door crossings with enhanced visibility
            door_color = '#FF7F00'  # Orange for heat map visibility
            door_width = 0.25
            
            if boundary['type'] == 'vertical':
                # Process vertical wall crossings
                wall_x = boundary['x']
                z_coords = [z for _, z in crossings]
                
                # Create histogram for successful path heat map
                hist, bin_edges = np.histogram(
                    z_coords, bins=50, range=(boundary['min_z'], boundary['max_z'])
                )
                
                # Apply smoothing for clean heat map appearance
                hist_smooth = gaussian_filter1d(hist, sigma=1.0)
                
                # Normalize intensities for visualization
                max_count = max(1, np.max(hist_smooth))
                normalized = hist_smooth / max_count
                
                # Draw enhanced heat map rectangles
                for i, intensity in enumerate(normalized):
                    if intensity > 0.05:  # Only draw if there's meaningful activity
                        z_start = bin_edges[i]
                        z_end = bin_edges[i+1]
                        
                        alpha = min(0.9, 0.2 + intensity * 0.8)  # Enhanced contrast
                        
                        rect = plt.Rectangle(
                            (wall_x - door_width/2, z_start),
                            door_width,
                            z_end - z_start,
                            facecolor=door_color,
                            edgecolor='none',
                            alpha=alpha,
                            zorder=7
                        )
                        ax.add_patch(rect)
                
            else:  # horizontal boundary
                # Process horizontal wall crossings
                wall_z = boundary['z']
                x_coords = [x for x, _ in crossings]
                
                # Create histogram for successful path heat map
                hist, bin_edges = np.histogram(
                    x_coords, bins=50, range=(boundary['min_x'], boundary['max_x'])
                )
                
                # Apply smoothing for clean heat map appearance
                hist_smooth = gaussian_filter1d(hist, sigma=1.0)
                
                # Normalize intensities for visualization
                max_count = max(1, np.max(hist_smooth))
                normalized = hist_smooth / max_count
                
                # Draw enhanced heat map rectangles
                for i, intensity in enumerate(normalized):
                    if intensity > 0.05:
                        x_start = bin_edges[i]
                        x_end = bin_edges[i+1]
                        
                        alpha = min(0.9, 0.2 + intensity * 0.8)
                        
                        rect = plt.Rectangle(
                            (x_start, wall_z - door_width/2),
                            x_end - x_start,
                            door_width,
                            facecolor=door_color,
                            edgecolor='none',
                            alpha=alpha,
                            zorder=7
                        )
                        ax.add_patch(rect)
    
    # Draw obstacle boxes with gray color for context
    if hasattr(env, 'boxes') and env.boxes:
        for box in env.boxes:
            # Extract position and size for 2D projection
            x, _, z = box.pos
            width, _, depth = box.size
            
            # Create rectangle centered at the box position
            rect = plt.Rectangle(
                (x - width/2, z - depth/2),  # Bottom-left corner
                width,                       # Width (x-dimension)
                depth,                       # Height (z-dimension)
                facecolor='#777777',         # Gray color for context
                edgecolor='#555555',         # Darker gray edge
                linewidth=0.5,
                alpha=0.5,
                zorder=4                     # Above floor, below paths
            )
            ax.add_patch(rect)
    
    # 2. Draw all paths without sampling for comprehensive view
    if paths and outcomes:
        print(f"Drawing all {len(paths)} paths without sampling...")
        
        # Draw failed paths first (background layer)
        failed_count = 0
        for i, (path, outcome) in enumerate(zip(paths, outcomes)):
            if not outcome:  # Failed paths
                failed_count += 1
                path_array = np.array(path)
                x, z = path_array[:, 0], path_array[:, 2]
                
                ax.plot(
                    x, z, 
                    color='red',
                    linewidth=1.0,  # Moderate thickness for failed paths
                    alpha=0.6,      # Semi-transparent
                    zorder=5
                )
                
                # Mark starting positions occasionally to reduce clutter
                if i % 10 == 0:  # Mark every 10th failed path starting point
                    ax.scatter(x[0], z[0], color='red', s=4, alpha=0.6, zorder=6)
        
        # Draw successful paths with enhanced prominence
        success_count = 0
        for i, (path, outcome) in enumerate(zip(paths, outcomes)):
            if outcome:  # Successful paths
                success_count += 1
                path_array = np.array(path)
                x, z = path_array[:, 0], path_array[:, 2]
                
                ax.plot(
                    x, z, 
                    color='blue',
                    linewidth=2.0,  # Thicker for successful paths to emphasize importance
                    alpha=0.75,     # More visible than failed paths
                    zorder=6
                )
                
                # Mark starting positions occasionally to avoid clutter
                if i % 5 == 0:  # Mark every 5th successful path starting point
                    ax.scatter(x[0], z[0], color='blue', s=6, alpha=0.8, zorder=7)
        
        print(f"Drew {failed_count} failed paths and {success_count} successful paths")
    
    # Mark terminal location prominently
    if hasattr(env, 'terminal_location'):
        term_x, term_z = env.terminal_location
        ax.scatter(
            term_x, term_z, 
            marker='*',
            s=140,
            color='green',
            edgecolor='darkgreen',
            linewidth=1.0,
            zorder=10,
            label='Terminal'
        )
    
    # Add informative title if not disabled
    if not no_title:
        # Create comprehensive title parts
        title_parts = []
        
        if iteration_num is not None:
            title_parts.append(f"Iteration {iteration_num}")
        
        if max_steps is not None:
            title_parts.append(f"Max Steps: {max_steps}")
            
        # Always include path analysis statistics
        title_parts.append(f"Paths: {successful_count}/{len(paths)} successful")
        
        if crossing_count > 0:
            title_parts.append(f"{crossing_count} crossings")
        
        # Combine title parts
        title = "Paths and Crossings - " + " - ".join(title_parts)
        
        # Add title to visualization
        ax.set_title(
            title,
            fontsize=11,
            color='black',
            pad=10
        )
    
    # Save with transparent background maintained
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename with analysis information
    iteration_part = f"_iter{iteration_num}" if iteration_num else ""
    steps_part = f"_steps{max_steps}" if max_steps else ""
    episodes_part = f"_ep{num_episodes}" if num_episodes else ""
    
    # Include success rate in filename for easy identification
    success_rate = (successful_count / len(paths) * 100) if paths else 0
    path_count_part = f"_s{successful_count}_f{len(paths)-successful_count}_{success_rate:.1f}pct"
    
    filename = f"{output_dir}/transparent_paths_success{iteration_part}{steps_part}{episodes_part}{path_count_part}_{timestamp}.png"
    
    # Save with high quality and transparency maintained
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved transparent paths visualization to: {filename}")
    plt.close(fig)
    
    return filename

def create_all_visualizations(env, paths=None, outcomes=None, 
                             output_dir="trajectories", show_all_paths=True,
                             iteration_num=None, max_steps=None, num_episodes=None):
    """
    Create all available visualization types in a coordinated batch process.
    
    This function generates multiple visualization types for comprehensive analysis:
    - Standard transparent paths
    - Success-only heat maps
    - Various academic and presentation formats
    
    Args:
        env: Environment object with world structure and dimensions
        paths: List of agent trajectory paths
        outcomes: List of path outcomes (True for success, False for failure)
        output_dir: Base directory for saving visualizations
        show_all_paths: Whether to include failed paths or only successful ones
        iteration_num: Training iteration number for file naming and titles
        max_steps: Maximum steps per episode for titles and context
        num_episodes: Number of episodes analyzed for titles and context
        
    Returns:
        Dictionary mapping visualization types to their saved file paths
    """
    import os
    
    # Create viz_layouts as a subfolder of the provided output_dir
    viz_dir = os.path.join(output_dir, "viz_layouts")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Count successful and failed paths for reporting
    if paths and outcomes:
        successful_count = sum(outcomes)
        failed_count = len(outcomes) - successful_count
        print(f"Creating visualizations for {len(paths)} paths: {successful_count} successful, {failed_count} failed")
    else:
        print("No paths provided for visualization")
        return {}
    
    result_files = {}
    
    # Filter paths if show_all_paths is False
    if not show_all_paths and paths and outcomes:
        print("Filtering to show only successful paths")
        filtered_paths = [path for path, outcome in zip(paths, outcomes) if outcome]
        filtered_outcomes = [True] * len(filtered_paths)
    else:
        filtered_paths = paths
        filtered_outcomes = outcomes
    
    # Create standard transparent paths visualization
    print("Creating standard transparent paths visualization...")
    result_files['transparent_paths'] = create_transparent_paths_visualization(
        env, filtered_paths, filtered_outcomes, viz_dir,
        no_title=False, width=15,
        iteration_num=iteration_num, max_steps=max_steps, num_episodes=num_episodes
    )
    
    # Create improved visualization with successful-only heat map
    print("Creating improved visualization with successful-only heat map...")
    result_files['successful_heat_map'] = create_transparent_paths_visualization_successful_only(
        env, paths, outcomes, viz_dir,  # Use all paths but calculate heat map from successful only
        no_title=False, width=15,
        iteration_num=iteration_num, max_steps=max_steps, num_episodes=num_episodes
    )
    
    print("\nAll visualizations created successfully!")
    for vis_type, filepath in result_files.items():
        print(f"- {vis_type}: {filepath}")
    
    return result_files

def parse_arguments():
    """
    Parse and validate command line arguments for the path visualizer.
    
    Provides comprehensive argument parsing for all visualization options,
    including model selection, environment configuration, and output customization.
    
    Returns:
        Parsed arguments namespace with all configuration options
    """
    parser = argparse.ArgumentParser(description='Create comprehensive path visualizations for multi-room escape environment analysis')
    
    # Model and environment configuration
    parser.add_argument('model_path', nargs='?', default=None, 
                       help='Path to the trained model file (.zip)')
    parser.add_argument('--env-file', type=str, default=None,
                       help='Path to environment file (default: auto-detect)')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Number of evaluation episodes to run (default: 100)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='trajectories',
                       help='Directory to save trajectory PNG files (default: trajectories)')
    
    # Visualization styling options
    parser.add_argument('--line-width', type=float, default=2.0,
                       help='Width of trajectory lines (default: 2.0)')
    parser.add_argument('--start-size', type=float, default=4,
                       help='Size of the markers at starting positions (default: 4)')
    parser.add_argument('--width', type=float, default=10,
                       help='Width of the output image in inches (default: 10)')
    
    # Path filtering and display options
    parser.add_argument('--successful-only', action='store_true',
                       help='Show only successful paths (hide failed attempts)')
    parser.add_argument('--no-title', action='store_true',
                       help='Omit the title from the visualization')
    
    # Color customization
    parser.add_argument('--successful-color', type=str, default='blue',
                       help='Color for successful paths (default: blue)')
    parser.add_argument('--failed-color', type=str, default='red',
                       help='Color for failed paths (default: red)')
    
    return parser.parse_args()

def main():
    """
    Main function that orchestrates the complete path visualization process.
    
    This function handles:
    1. Argument parsing and validation
    2. Model and environment loading
    3. Episode evaluation and data collection
    4. Multiple visualization generation
    5. Results reporting and file management
    
    The main workflow:
    - Loads a trained RL model
    - Runs evaluation episodes to collect trajectory data
    - Generates comprehensive visualizations for analysis
    - Provides statistical summaries of agent performance
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Extract iteration number from model path for tracking purposes
    iteration_num = None
    if args.model_path:
        # Look for patterns like "iteration_10" in the model path
        import re
        match = re.search(r'iteration_(\d+)', args.model_path)
        if match:
            iteration_num = match.group(1)
    
    # Find and validate environment file
    env_file = args.env_file or find_environment_file()
    print(f"Using environment file: {env_file}")
    
    # Import environment module dynamically
    env_module = import_environment_module(env_file)
    
    # Get necessary classes from the imported module
    CustomEnv = getattr(env_module, 'CustomEnv')
    RobustNoRenderEnv = getattr(env_module, 'RobustNoRenderEnv')
    
    # Find and validate model file
    model_path = find_model_path(args.model_path)
    if not model_path:
        print("ERROR: Could not find a model file (.zip)")
        print("Please specify the model path as a command line argument")
        return
    
    # Extract iteration number from found model path if not already found
    if iteration_num is None and model_path:
        import re
        match = re.search(r'iteration_(\d+)', model_path)
        if match:
            iteration_num = match.group(1)
    
    print(f"Loading model from: {model_path}")
    print(f"Iteration number detected: {iteration_num or 'None'}")
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Create the environment with appropriate parameters
    max_episode_steps = 500  # Default value for evaluation
    env = RobustNoRenderEnv(
        max_episode_steps=max_episode_steps,
        steps_until_hallway=500,
        reward_scales={
            # Default reward scaling configuration
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
    )
    
    # Store environment attributes for analysis
    world_width = env.world_width
    world_depth = env.world_depth
    print(f"Environment dimensions: {world_width} x {world_depth}")
    
    # Initialize tracking attributes for data collection
    env.successful_paths = []
    env.all_paths = []
    env.all_outcomes = []
    
    door_positions = []  # Track door creation events
    
    # Run comprehensive evaluation
    print(f"\nRunning {args.num_episodes} evaluation episodes...")
    start_time = time.time()
    
    try:
        for episode in range(args.num_episodes):
            # Reset environment with random seed for variety
            observation, info = env.reset(seed=random.randint(0, 10000))
            
            # Print progress periodically for user feedback
            if args.num_episodes <= 20 or episode % 10 == 0:
                print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            
            # Initialize path tracking for this episode
            current_path = [env.agent.pos.copy()]
            
            # Run episode until completion
            for step in range(env.max_episode_steps):
                # Get action from trained model
                action, _states = model.predict(observation, deterministic=True)
                
                # Execute action in environment
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Track agent trajectory
                current_path.append(env.agent.pos.copy())

                # Track door creation events for analysis
                if info.get('created_door', False):
                    door_pos = env.agent.pos.copy()
                    door_positions.append(door_pos)
                    print(f"Door created at position: {door_pos}")
                
                # Check for episode termination
                if terminated or truncated:
                    step_count = step + 1
                    reached_terminal = info.get('reached_terminal', False) or getattr(env, 'reached_terminal_area', False)
                    
                    # Save trajectory data for analysis
                    env.all_paths.append(current_path)
                    env.all_outcomes.append(reached_terminal)
                    
                    # Report episode outcome
                    if args.num_episodes <= 20 or episode % 10 == 0:
                        if reached_terminal:
                            print(f"  ✓ Success! Reached terminal state in {step_count} steps")
                        else:
                            print(f"  ✗ Failed to reach terminal state")
                    
                    # Track successful paths separately
                    if step_count < 500:
                        env.successful_paths.append(current_path)
                    
                    break
                    
        # Calculate evaluation statistics
        elapsed_time = time.time() - start_time
        
        # Generate comprehensive statistics report
        success_count = len(env.successful_paths)
        success_rate = success_count / args.num_episodes * 100
        print(f"\nEvaluation complete in {elapsed_time:.2f} seconds!")
        print(f"Success rate: {success_rate:.1f}% ({success_count}/{args.num_episodes})")
        
        # Create comprehensive visualizations
        print("\nCreating all visualizations...")
        create_all_visualizations(
            env, 
            paths=env.all_paths,
            outcomes=env.all_outcomes,
            output_dir=args.output_dir,
            show_all_paths=not args.successful_only,
            iteration_num=iteration_num,
            max_steps=env.max_episode_steps,
            num_episodes=args.num_episodes
        )
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        print("Evaluation completed")


if __name__ == "__main__":
    """
    Entry point for the path visualization script.
    
    This script provides comprehensive analysis tools for multi-room escape environment
    training results. It loads trained RL models and generates various visualization
    types to understand agent behavior, learning progress, and strategy effectiveness.
    
    Key Features:
    - Automatic model and environment detection
    - Multiple visualization types for different analysis needs
    - Comprehensive statistical reporting
    - Configurable output options for different use cases
    - Support for academic and presentation-ready outputs
    
    The script is designed to work with checkpoints from the training process,
    automatically extracting relevant metadata like iteration numbers for
    comprehensive analysis workflows.
    """
    main()