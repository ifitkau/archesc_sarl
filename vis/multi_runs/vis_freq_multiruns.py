"""
Multi-Agent Reinforcement Learning Visualization Script - Picture-in-Picture Analysis

This script creates comprehensive visualizations for multi-agent reinforcement learning
training runs, featuring a main plot with an embedded picture-in-picture zoom view
for detailed analysis of specific return value ranges.

Key Features:
- Multi-agent data comparison with distinct styling
- True picture-in-picture inset plot with connecting lines
- Correlation analysis between agents (when exactly 2 agents)
- Customizable zoom region for detailed analysis
- Colorblind-friendly color palette
- Optional seaborn integration for enhanced aesthetics
- Configurable steps-per-iteration conversion
- Statistical summary reporting

Use Cases:
- Comparing navigator vs door controller performance
- Analyzing convergence patterns in multi-agent systems
- Detailed examination of high-return regions
- Performance correlation analysis between agents

Author: Research Team
Date: 2025
License: MIT
Dependencies: matplotlib, numpy, optional: seaborn
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch
from datetime import datetime

# Try to import seaborn for better plot aesthetics
# Seaborn provides improved default styling and color palettes
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: Install seaborn for better plot aesthetics: pip install seaborn")

# Configuration constants
OUTPUT_DIR = "multi_agent_figures"  # Default output directory for generated plots
DPI = 300                           # High resolution for publication-quality figures
STEPS_PER_ITERATION = 12000         # Default conversion factor from steps to iterations
UPDATE_INTERVAL = 2048              # RL algorithm update frequency (for reference)

# Color palettes specifically designed for colorblind accessibility
# These colors are distinguishable for users with various forms of color vision deficiency
COLORBLIND_PALETTE = {
    "navigator": "#0072B2",         # Blue - for navigator agent
    "door_controller": "#C185A2",   # Pink/Magenta - for door controller agent
}

class AgentData:
    """
    Class to store and process data from a single reinforcement learning agent
    
    This class handles loading, processing, and statistical analysis of training data
    for individual agents in a multi-agent reinforcement learning system.
    
    Attributes:
        file_path (str): Path to the data file
        agent_name (str): Internal name identifier for the agent
        label (str): Human-readable label for plots and legends
        color (str): Hex color code for plotting this agent's data
        steps (np.array): Raw step numbers from training
        values (np.array): Return/reward values corresponding to steps
        iterations (np.array): Steps converted to iterations using STEPS_PER_ITERATION
        mean (float): Overall mean return value
        median (float): Overall median return value
        min (float): Minimum return value achieved
        max (float): Maximum return value achieved
        std (float): Standard deviation of return values
    """
    
    def __init__(self, file_path, agent_name):
        """
        Initialize an AgentData object
        
        Args:
            file_path (str): Path to the JSON file containing agent training data
            agent_name (str): Identifier for the agent (e.g., 'navigator', 'door_controller')
        """
        self.file_path = file_path
        self.agent_name = agent_name
        # Convert agent name to a human-readable label (replace underscores, capitalize)
        self.label = agent_name.replace('_', ' ').title()
        # Assign color from palette, default to black if agent not in palette
        self.color = COLORBLIND_PALETTE.get(agent_name, "#000000")
        
        # Load and process the training data
        self.steps, self.values = self.load_data()
        
        # Convert raw steps to iterations for more meaningful x-axis scaling
        self.iterations = self.steps / STEPS_PER_ITERATION
        
        # Calculate comprehensive statistics for analysis and reporting
        self.mean = np.mean(self.values)
        self.median = np.median(self.values)
        self.min = np.min(self.values)
        self.max = np.max(self.values)
        self.std = np.std(self.values)
    
    def load_data(self):
        """
        Load training data from JSON file
        
        Supports JSON format with list of entries containing:
        [timestamp, step_number, return_value]
        
        Returns:
            tuple: (steps_array, values_array) as numpy arrays
            
        Raises:
            ValueError: If the file format is not supported or cannot be parsed
        """
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        if file_ext == '.json':
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Handle JSON format: list of [timestamp, steps, value] entries
            if isinstance(data, list):
                steps = []
                values = []
                for entry in data:
                    if isinstance(entry, list) and len(entry) >= 3:
                        # Format: [[timestamp, steps, value], ...]
                        # Extract step number (index 1) and return value (index 2)
                        steps.append(entry[1])
                        values.append(entry[2])
                return np.array(steps), np.array(values)
        
        # If we reach here, the file format is not supported
        raise ValueError(f"Could not parse data from {self.file_path}")
    
    def print_statistics(self):
        """
        Print comprehensive statistics for this agent's performance
        
        Outputs a formatted summary including:
        - Data point count
        - Central tendency measures (mean, median)
        - Spread measures (min, max, standard deviation)
        - Training duration (iteration range)
        """
        print(f"\n===== {self.label} Return Statistics =====")
        print(f"Data points: {len(self.values)}")
        print(f"Overall mean: {self.mean:.2f}")
        print(f"Overall median: {self.median:.2f}")
        print(f"Overall min: {self.min:.2f}")
        print(f"Overall max: {self.max:.2f}")
        print(f"Overall std dev: {self.std:.2f}")
        print(f"Iteration range: {self.iterations[0]:.1f} to {self.iterations[-1]:.1f}")


def create_true_picture_in_picture_plot(agents, output_path, inset_y_min=450, inset_y_max=510, inset_size=(0.5, 0.5), max_iteration=1000):
    """
    Create a comprehensive plot with picture-in-picture zoom functionality
    
    This function generates a main plot showing the full training curves for all agents,
    with an embedded inset plot that zooms into a specific return value range for
    detailed analysis. Connection lines clearly indicate the relationship between
    the main plot and the zoomed region.
    
    Args:
        agents (list): List of AgentData objects to plot
        output_path (str): File path where the plot will be saved
        inset_y_min (float): Minimum return value for the zoom inset
        inset_y_max (float): Maximum return value for the zoom inset
        inset_size (tuple): (width, height) as fractions of figure size for inset
        max_iteration (int): Maximum iteration to display on x-axis
        
    Features:
        - Main plot with full training curves and mean lines
        - Picture-in-picture inset with zoomed view of specified return range
        - Connection lines showing the relationship between plots
        - Correlation analysis for two-agent scenarios
        - Professional formatting with grids and labels
    """
    # Create figure and main axes with publication-quality size
    fig, ax_main = plt.subplots(figsize=(16, 10))
    
    # Plot each agent's training curve on the main plot
    for agent in agents:
        # Plot raw training data with transparency to show overlapping points
        ax_main.plot(agent.iterations, agent.values, 
                 color=agent.color, alpha=0.7, linewidth=1.2, 
                 label=f"{agent.label} (raw)")
        
        # Add horizontal reference line for mean performance
        ax_main.axhline(agent.mean, linestyle='--', color=agent.color, 
                     alpha=0.7, linewidth=1.5,
                     label=f"{agent.label} Mean: {agent.mean:.1f}")
    
    # Format main plot with professional styling
    ax_main.set_xlabel('Iteration', fontsize=14)
    ax_main.set_ylabel('Return', fontsize=14)
    ax_main.set_title('Comparison of Environment Types - Agent Returns', fontsize=16)
    ax_main.grid(True, alpha=0.3, linestyle=':')
    
    # Set explicit x-axis limits to ensure consistent scaling
    ax_main.set_xlim(left=0, right=max_iteration)
    
    # Create custom formatter to display clean integer values (no commas for readability)
    def plain_int_formatter(x, pos):
        """Format tick labels as plain integers without thousand separators"""
        return f"{int(x)}"
    
    # Apply the custom formatter to both axes for clean appearance
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(plain_int_formatter))
    
    # Create inset axis manually for precise control over positioning
    # Position coordinates are [left, bottom, width, height] in figure coordinates (0-1)
    width, height = inset_size
    # Center the inset horizontally and vertically within the plot area
    left = 0.5 - width/2.5  # Slight offset from center for better visual balance
    bottom = 0.5 - height/2
    ax_inset = fig.add_axes([left, bottom, width, height])
    
    # Plot the same data on the inset but with lighter styling
    for agent in agents:
        # Use thinner lines in the inset to avoid visual clutter
        ax_inset.plot(agent.iterations, agent.values, 
                  color=agent.color, alpha=0.7, linewidth=1.0)
        
        # Include mean reference lines in the inset as well
        ax_inset.axhline(agent.mean, linestyle='--', color=agent.color, 
                      alpha=0.7, linewidth=1.0)
    
    # Configure the inset to focus on the specified return value range
    ax_inset.set_ylim(inset_y_min, inset_y_max)
    ax_inset.set_xlim(left=0, right=max_iteration)
    
    # Format inset plot with appropriate labels and styling
    ax_inset.grid(True, alpha=0.3, linestyle=':')
    ax_inset.set_ylabel('Return', fontsize=12, rotation=90, labelpad=5)
    ax_inset.set_xlabel('Iteration', fontsize=12)
    ax_inset.set_title('Zoomed View', fontsize=12)
    
    # Apply consistent number formatting to inset axes
    ax_inset.xaxis.set_major_formatter(ticker.FuncFormatter(plain_int_formatter))
    
    # Create visual connection between main plot and inset using connection lines
    # Define the corners of the zoom region in data coordinates
    inset_corners = [
        (0, inset_y_min),              # bottom left corner
        (max_iteration, inset_y_min),  # bottom right corner
        (max_iteration, inset_y_max),  # top right corner
        (0, inset_y_max)               # top left corner
    ]
    
    # Define corresponding points in the main plot (same coordinates)
    main_corners = [
        (0, inset_y_min),              # bottom left corner
        (max_iteration, inset_y_min),  # bottom right corner
        (max_iteration, inset_y_max),  # top right corner
        (0, inset_y_max)               # top left corner
    ]
    
    # Draw connection lines between each corner of the inset and main plot
    for i in range(4):
        # Create a connection patch for each corner
        con = ConnectionPatch(
            xyA=inset_corners[i], xyB=main_corners[i],
            coordsA="data", coordsB="data",
            axesA=ax_inset, axesB=ax_main,
            color="lightgray", linewidth=1, alpha=0.6
        )
        # Add the connection patch to the inset axes
        ax_inset.add_artist(con)
    
    # Add legend to main plot for agent identification
    ax_main.legend(loc='lower right', fontsize=10)
    
    # Apply tight layout to optimize spacing and prevent label cutoff
    plt.tight_layout()
    
    # Ensure output directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved true picture-in-picture plot with connecting lines to {output_path}")
    
    # Calculate and report correlation for two-agent scenarios
    if len(agents) == 2:
        # Find overlapping data points for correlation calculation
        # (in case agents have different training lengths)
        min_len = min(len(agents[0].values), len(agents[1].values))
        # Calculate Pearson correlation coefficient
        corr = np.corrcoef(agents[0].values[:min_len], agents[1].values[:min_len])[0, 1]
        print(f"Correlation between {agents[0].label} and {agents[1].label}: {corr:.4f}")
    
    # Clean up memory by closing the figure
    plt.close()


def main():
    """
    Main function handling command-line interface and orchestrating the visualization process
    
    Parses command-line arguments, loads agent data, generates statistics,
    and creates the picture-in-picture visualization plot.
    """
    # Set up comprehensive command-line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Agent RL Visualization Tool')
    
    # Required input files for the two agents
    parser.add_argument('--navigator-reward', type=str, required=True,
                        help='Navigator reward data file (JSON format)')
    parser.add_argument('--door-reward', type=str, required=True,
                        help='Door controller reward data file (JSON format)')
    
    # Optional output configuration arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (if not specified, uses default naming)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for generated plots')
    
    # Optional agent labeling arguments for custom display names
    parser.add_argument('--navigator-label', type=str, default=None,
                     help='Custom label for navigator data in plots and legends')
    parser.add_argument('--door-label', type=str, default=None,
                     help='Custom label for door controller data in plots and legends')
    
    # Picture-in-picture configuration arguments
    parser.add_argument('--inset-min', type=float, default=450,
                     help='Minimum return value for inset y-axis (zoom lower bound)')
    parser.add_argument('--inset-max', type=float, default=510,
                     help='Maximum return value for inset y-axis (zoom upper bound)')
    parser.add_argument('--inset-width', type=float, default=0.5,
                     help='Width of inset as fraction of figure size (0.0-1.0)')
    parser.add_argument('--inset-height', type=float, default=0.5,
                     help='Height of inset as fraction of figure size (0.0-1.0)')
    
    # Data processing configuration arguments
    parser.add_argument('--steps-per-iteration', type=int, default=12000,
                     help='Number of environment steps per training iteration')
    parser.add_argument('--max-iteration', type=int, default=1000,
                     help='Maximum iteration value to display on x-axis')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Update global configuration with user-specified steps per iteration
    global STEPS_PER_ITERATION
    STEPS_PER_ITERATION = args.steps_per_iteration
    
    # Load and initialize agent data objects
    agents = []
    
    # Create navigator agent data object
    navigator_agent = AgentData(args.navigator_reward, "navigator")
    if args.navigator_label:
        # Override default label if custom label provided
        navigator_agent.label = args.navigator_label
    agents.append(navigator_agent)
    
    # Create door controller agent data object
    door_agent = AgentData(args.door_reward, "door_controller")
    if args.door_label:
        # Override default label if custom label provided
        door_agent.label = args.door_label
    agents.append(door_agent)
    
    # Generate and display statistical summaries for each agent
    for agent in agents:
        agent.print_statistics()
    
    # Determine output file path (use default if not specified)
    output_path = args.output or os.path.join(
        args.output_dir, 
        "multi_agent_returns_true_pic_in_pic.png"
    )
    
    # Create the main visualization with all specified parameters
    create_true_picture_in_picture_plot(
        agents, 
        output_path, 
        inset_y_min=args.inset_min, 
        inset_y_max=args.inset_max,
        inset_size=(args.inset_width, args.inset_height),
        max_iteration=args.max_iteration
    )


# Entry point for script execution
if __name__ == "__main__":
    main()