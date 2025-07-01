"""
Multi-Run Episode Length Analysis and Visualization

This script creates comprehensive visualizations for analyzing and comparing episode length
data across multiple reinforcement learning experiments. It processes WandB export CSV files
and generates plots with segment analysis, moving averages, and picture-in-picture zoom views.

Key Features:
- Supports multiple experiment runs comparison
- Moving average calculation with configurable window size
- Segment-based statistical analysis
- Picture-in-picture inset for detailed view of specific value ranges
- Colorblind-friendly color palette
- Customizable plot parameters (axis limits, segment count, etc.)

Date: July 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import ConnectionPatch

# Configuration constants for the visualization
WINDOW_SIZE = 10  # Moving average window size - number of data points to average
SEGMENT_COUNT = 5  # Number of segments to divide the data into for analysis
DPI = 300  # Image resolution for saved plots
# Colorblind-friendly palette - ensures accessibility for users with color vision deficiencies
COLORBLIND_PALETTE = ['#0173B2', '#C185A2', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']

class RunData:
    """
    Class to hold and process data for a single experimental run
    
    This class encapsulates all data and methods needed to process a single run's
    episode length data, including moving average calculations and segment analysis.
    
    Attributes:
        label (str): Human-readable label for this run
        values (np.array): Episode length values
        steps (np.array): Iteration/step numbers corresponding to values
        color (str): Color hex code for plotting this run
        moving_average_values (np.array): Calculated moving average values
        segments (list): List of dictionaries containing segment statistics
    """
    def __init__(self, label, values=None, steps=None, color=None):
        """
        Initialize a RunData object
        
        Args:
            label (str): Name/label for this run
            values (array-like, optional): Episode length values
            steps (array-like, optional): Step/iteration numbers
            color (str, optional): Hex color code for plotting
        """
        self.label = label
        self.values = values if values is not None else []
        self.steps = steps if steps is not None else []
        self.color = color
        self.moving_average_values = None
        self.segments = []
    
    def calculate_moving_average(self, window_size=WINDOW_SIZE):
        """
        Calculate moving average with given window size
        
        Uses scipy's uniform_filter1d for efficient calculation. The 'reflect' mode
        pads edges to avoid NaN values at the beginning of the series.
        
        Args:
            window_size (int): Number of points to include in moving average window
        """
        if len(self.values) > 0:
            # Use uniform filter for more efficient moving average calculation
            # Pad the edges to avoid NaN values at the beginning
            self.moving_average_values = uniform_filter1d(self.values, size=window_size, mode='reflect')
    
    def analyze_segments(self, num_segments=SEGMENT_COUNT):
        """
        Divide the data into segments and calculate statistics for each
        
        Segments are created by dividing the step range into equal parts.
        For each segment, calculates mean, standard deviation, and moving average mean.
        
        Args:
            num_segments (int): Number of segments to create
        """
        if len(self.steps) == 0 or len(self.values) == 0:
            return
        
        # Define segment boundaries - equal divisions by step value
        max_step = max(self.steps)
        segment_size = max_step / num_segments
        
        self.segments = []
        for i in range(num_segments):
            start_step = i * segment_size
            end_step = (i + 1) * segment_size if i < num_segments - 1 else max_step
            
            # Find indices of steps within this segment
            indices = np.where((self.steps >= start_step) & (self.steps < end_step))[0]
            
            # Calculate statistics for this segment
            if len(indices) > 0:
                segment_values = self.values[indices]
                segment_mean = np.mean(segment_values)
                segment_std = np.std(segment_values)
                segment_ma_values = self.moving_average_values[indices] if self.moving_average_values is not None else []
                segment_ma_mean = np.mean(segment_ma_values) if len(segment_ma_values) > 0 else np.nan
            else:
                segment_mean = segment_std = segment_ma_mean = np.nan
            
            # Store segment information
            self.segments.append({
                'start_step': start_step,
                'end_step': end_step,
                'mean': segment_mean,
                'std': segment_std,
                'ma_mean': segment_ma_mean
            })


def parse_csv_data(csv_file, exp_id=None):
    """
    Parse the WandB CSV export file to extract episode length data
    
    Handles both semicolon-separated and standard comma-separated CSV formats.
    Automatically detects experiment IDs from column names and extracts relevant data.
    
    Args:
        csv_file (str): Path to the WandB export CSV file
        exp_id (str, optional): Specific experiment ID to extract (if None, use default list)
        
    Returns:
        tuple: (DataFrame, list of run columns, list of all experiment IDs found)
    """
    try:
        # Read the CSV data - need to handle the semicolon-separated format
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Read the first line to determine format
            header_line = f.readline().strip()
            
        # Determine CSV separator format
        if ';' in header_line:
            # Read it with pandas using semicolon as separator
            df = pd.read_csv(csv_file, sep=';')
        else:
            # Standard CSV format
            df = pd.read_csv(csv_file)
        
        # Standardize column names - WandB sometimes uses 'global_step' instead of 'Step'
        if 'global_step' in df.columns:
            df.rename(columns={'global_step': 'Step'}, inplace=True)
        
        # Default experiment IDs to look for - these are typical WandB run IDs
        default_exp_ids = ['wh6zievy', 'ggsmin6d', 'yek7ws5j', 'afhgtkn0', 'mpvfa69w', '4s4w8dq1']
        
        # Get column names for the specified experiment ID or all default ones
        if exp_id:
            run_columns = [col for col in df.columns if col.startswith(exp_id)]
        else:
            run_columns = [col for col in df.columns if any(col.startswith(exp_id) for exp_id in default_exp_ids)]
        
        # Extract all experiment IDs from column names for reference
        all_exp_ids = set()
        for col in df.columns:
            parts = col.split('_')
            if len(parts) > 0 and parts[0] not in ['Step', 'global', '']:
                all_exp_ids.add(parts[0])
        
        # Debug output to help users understand what data was found
        print(f"Found experiment IDs: {all_exp_ids}")
        print(f"Found columns: {run_columns}")
        
        return df, run_columns, list(all_exp_ids)
    
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return None, [], []


def parse_iteration_data(txt_file):
    """
    Parse the iteration data from the text file
    
    Parses a text file containing iteration data in CSV format with columns:
    iteration, episodes, terminals, rewards
    
    Args:
        txt_file (str): Path to the iteration data text file
        
    Returns:
        dict: Dictionary with arrays for iterations, episodes, terminals, rewards
        None: If parsing fails
    """
    try:
        iterations = []
        episodes = []
        terminals = []
        rewards = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip header lines and empty lines
                if line.startswith('---') or not line or line.startswith('Total'):
                    continue
                    
                parts = line.split(',')
                # Ensure we have the expected number of columns and first column is numeric
                if len(parts) >= 4 and parts[0].isdigit():
                    iterations.append(int(parts[0]))
                    episodes.append(int(parts[1]))
                    terminals.append(int(parts[2]))
                    rewards.append(float(parts[3]))
        
        return {
            'iterations': np.array(iterations),
            'episodes': np.array(episodes),
            'terminals': np.array(terminals),
            'rewards': np.array(rewards)
        }
    
    except Exception as e:
        print(f"Error parsing iteration data file: {e}")
        return None


def create_multi_run_plot(csv_files, txt_file, output_path, run_names=None, run_descriptions=None, 
                       exp_ids=None, window_size=WINDOW_SIZE, num_segments=SEGMENT_COUNT, 
                       custom_vlines=None, x_max=1000, y_max=510, show_ma=True, show_raw=True,
                       inset_y_min=450, inset_y_max=510, inset_size=(0.5, 0.5)):
    """
    Create a plot analyzing episode length over iterations for multiple runs
    
    This is the main function that creates a comprehensive visualization with:
    - Main plot showing all runs with moving averages and/or raw data
    - Segment analysis with mean values displayed
    - Picture-in-picture inset showing zoomed view of specified y-range
    - Customizable vertical lines for marking important iterations
    
    Args:
        csv_files (list): List of paths to the WandB export CSV files
        txt_file (str): Path to the iteration data text file
        output_path (str): Path to save the output image
        run_names (list, optional): List of run names (e.g., "Run 1", "Run 2")
        run_descriptions (list, optional): List of run descriptions (e.g., "Terminal Right", "Terminal Left")
        exp_ids (list, optional): List of experiment IDs to extract
        window_size (int): Window size for moving average
        num_segments (int): Number of segments for analysis
        custom_vlines (list, optional): List of x-coordinates for custom vertical lines
        x_max (int): Maximum value for x-axis
        y_max (int): Maximum value for y-axis
        show_ma (bool): Whether to show moving average lines
        show_raw (bool): Whether to show raw data lines
        inset_y_min (float): Minimum value for inset y-axis
        inset_y_max (float): Maximum value for inset y-axis
        inset_size (tuple): Tuple of (width, height) for inset as fraction of figure size
    """
    # Ensure csv_files is a list even if single file is passed
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Generate default run names if not provided
    if run_names is None:
        run_names = [f"Run {i+1}" for i in range(len(csv_files))]
    elif len(run_names) < len(csv_files):
        # Extend run_names if needed
        run_names.extend([f"Run {i+len(run_names)+1}" for i in range(len(csv_files) - len(run_names))])
    
    # Generate default run descriptions if not provided
    if run_descriptions is None:
        run_descriptions = [""] * len(csv_files)
    elif len(run_descriptions) < len(csv_files):
        # Extend run_descriptions with empty strings if needed
        run_descriptions.extend([""] * (len(csv_files) - len(run_descriptions)))
    
    # Generate default exp_ids to None for all files if not provided
    if exp_ids is None:
        exp_ids = [None] * len(csv_files)
    elif len(exp_ids) < len(csv_files):
        # Extend exp_ids with None if needed
        exp_ids.extend([None] * (len(csv_files) - len(exp_ids)))
    
    # Parse iteration data (currently required but not used in main plot)
    iter_data = parse_iteration_data(txt_file)
    if iter_data is None:
        return
    
    # Process all runs and create RunData objects
    runs = []
    
    for i, (csv_file, run_name, run_desc, exp_id) in enumerate(zip(csv_files, run_names, run_descriptions, exp_ids)):
        # Parse CSV data for this run
        df, run_columns, all_exp_ids = parse_csv_data(csv_file, exp_id)
        if df is None:
            continue
        
        # If exp_id is None, try to use the first experiment ID found
        if exp_id is None and len(all_exp_ids) > 0:
            exp_id = list(all_exp_ids)[0]
            print(f"Using experiment ID: {exp_id} for {run_name}")
        
        # Find a column for this experiment - be more specific about column selection
        run_column = None
        
        # If we have a specific exp_id, look for columns that match exactly
        if exp_id:
            exact_match_columns = [col for col in run_columns if col.startswith(exp_id + '_')]
            # If we have an exact match, use it
            if exact_match_columns:
                run_column = exact_match_columns[0]
                print(f"Found matching column for {exp_id}: {run_column}")
            # Otherwise, try less strict matching
            else:
                run_column = next((col for col in run_columns if col.startswith(exp_id)), None)
                if run_column:
                    print(f"Found column starting with {exp_id}: {run_column}")
        
        # If still no column found and we have available columns, use the first one
        if not run_column and run_columns:
            run_column = run_columns[0]
            print(f"Using first available column: {run_column} for {run_name}")
        
        if run_column is None:
            print(f"Could not find data column for {run_name}")
            continue
        
        # Extract steps and values from the dataframe
        steps = df['Step'].values // 12000  # Convert steps to iterations (assuming 12000 steps per iteration)
        values = df[run_column].values
        
        # Create full label combining name and description
        if run_desc:
            full_label = f"{run_name} - {run_desc}"
        else:
            full_label = run_name
        
        # Create RunData object with assigned color from palette
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        run = RunData(label=full_label, values=values, steps=steps, color=color)
        
        # Calculate moving average for this run
        run.calculate_moving_average(window_size)
        
        # Analyze segments for this run
        run.analyze_segments(num_segments)
        
        runs.append(run)
    
    # Check if we have any valid run data
    if not runs:
        print("No valid run data found")
        return
    
    # Create the figure with the main plot
    fig = plt.figure(figsize=(14, 9))
    
    # Create main axis using subplot2grid for future extensibility
    ax_main = plt.subplot2grid((1, 1), (0, 0))
    
    # Calculate segment boundaries based on x_max for consistent segmentation
    segment_size = x_max / num_segments
    segments = []
    
    for i in range(num_segments):
        start_step = int(i * segment_size)
        end_step = int((i + 1) * segment_size) if i < num_segments - 1 else x_max
        
        segments.append({
            'start_step': start_step,
            'end_step': end_step,
            'label': f"Segment {i+1}\n{start_step}-{end_step}"
        })
    
    # Set precise axis limits for main plot
    ax_main.set_xlim(0, x_max)
    ax_main.set_ylim(0, y_max)
    
    # Add segment boundaries and labels on main plot
    for i, segment in enumerate(segments):
        # Add vertical line for segment boundary (skip first one at x=0)
        if i > 0:  # Skip first segment start
            ax_main.axvline(segment['start_step'], color='gray', linestyle='--', alpha=0.3)
        
        # Add segment boundary label with positioning logic
        label_x = segment['start_step'] if i > 0 else 0
        
        # Offset label position to avoid overlap with vertical lines
        if i == 0:
            label_x += 200
        
        if i:
            label_x += 200
        
        # Place segment labels at top of plot
        ax_main.text(label_x, ax_main.get_ylim()[1] * 0.98, 
                f"Segment {i+1}\n{segment['start_step']}-{segment['end_step']}", 
                rotation=90, va='top', ha='right', 
                color='gray', alpha=0.7, fontsize=8)
    
    # Add custom vertical lines if specified by user
    if custom_vlines:
        for vline_x in custom_vlines:
            if 0 <= vline_x <= x_max:  # Ensure line is within plot bounds
                ax_main.axvline(vline_x, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
                
                # Add a small label for the custom line
                ax_main.text(vline_x, ax_main.get_ylim()[1] * 0.9, 
                        f"Line at {vline_x}", 
                        rotation=90, va='top', ha='right', 
                        color='gray', alpha=0.9, fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    # Determine segment ranges for analysis (custom vlines override default segments)
    if custom_vlines and len(custom_vlines) >= 2:
        # Sort the custom vlines and create segments based on them
        sorted_vlines = sorted([0] + custom_vlines + [x_max])
        
        # Create segments from pairs of adjacent vlines
        segment_ranges = []
        for i in range(len(sorted_vlines) - 1):
            start = sorted_vlines[i]
            end = sorted_vlines[i+1]
            segment_ranges.append((i+1, start, end))
    else:
        # Use default segment boundaries
        segment_size = x_max / num_segments
        segment_ranges = []
        for i in range(num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size
            segment_ranges.append((i+1, start, end))
    
    # Plot each run on main plot
    for i, run in enumerate(runs):
        # Plot raw data if enabled
        if show_raw:
            ax_main.plot(run.steps, run.values, color=run.color, alpha=0.7, linewidth=1.0,
                    label=f"{run.label} (Raw Data)")
        
        # Plot moving average if enabled
        if show_ma:
            ax_main.plot(run.steps, run.moving_average_values, color=run.color, linewidth=2.0, 
                    label=f"{run.label} (MA={window_size})")
        
        # Add horizontal line for overall mean
        overall_mean = np.mean(run.values)
        ax_main.axhline(overall_mean, linestyle=':', color=run.color, alpha=0.5,
                  label=f"{run.label} Mean: {overall_mean:.1f}")
        
        # Calculate means for each segment
        segment_means = []
        for seg_id, start, end in segment_ranges:
            # Find values within this range
            indices = np.where((run.steps >= start) & (run.steps < end))[0]
            if len(indices) > 0:
                # Use the raw values for mean calculation if moving average is not shown
                if show_ma:
                    segment_values = run.moving_average_values[indices]
                else:
                    segment_values = run.values[indices]
                    
                valid_values = segment_values[~np.isnan(segment_values)]
                segment_mean = np.mean(valid_values) if len(valid_values) > 0 else np.nan
                segment_means.append((seg_id, start, end, segment_mean))
        
        # Draw segment means on main plot
        for seg_id, start, end, mean in segment_means:
            if np.isnan(mean):
                continue
                
            # Horizontal line for segment mean
            ax_main.hlines(mean, start, end, colors=run.color, linestyles='--', alpha=1.0, linewidth=1.5)
            
            # Center point for the segment
            mid_point = (start + end) / 2
            
            # Add dot at the mean point
            ax_main.plot(mid_point, mean, 'o', color=run.color, markersize=5, alpha=1.0)
            
            # Add text box with mean value - staggered by run index
            # This helps avoid overlapping labels for multiple runs
            offset = i * 0.05 * y_max  # Offset each run's labels
            y_pos = ax_main.get_ylim()[1] * (0.85 if seg_id > 0 else 0.4) - offset
            
            ax_main.text(mid_point, y_pos, 
                    f"{run.label}: {mean:.1f}", 
                    ha='center', va='center', fontsize=9, color=run.color,
                    bbox=dict(facecolor='white', edgecolor=run.color, alpha=1.0, 
                             boxstyle='round,pad=0.3'),
                    zorder=10)
    
    # Create inset axis manually for the zoomed-in view (picture-in-picture)
    # Position is [left, bottom, width, height] in figure coordinates
    width, height = inset_size
    left = 0.3  # Position from left edge of figure
    bottom = 0.35  # Position from bottom edge of figure
    ax_inset = fig.add_axes([left, bottom, width, height])
    
    # Plot each run on the inset with same styling but reduced line weights
    for i, run in enumerate(runs):
        # Plot raw data if enabled (lighter weight for inset)
        if show_raw:
            ax_inset.plot(run.steps, run.values, color=run.color, alpha=0.5, linewidth=0.8)
        
        # Plot moving average if enabled (lighter weight for inset)
        if show_ma:
            ax_inset.plot(run.steps, run.moving_average_values, color=run.color, linewidth=1.5)
        
        # Add horizontal line for overall mean
        overall_mean = np.mean(run.values)
        ax_inset.axhline(overall_mean, linestyle=':', color=run.color, alpha=0.5)
    
    # Set y-axis limits for inset plot to focus on the high values region
    ax_inset.set_ylim(inset_y_min, inset_y_max)
    ax_inset.set_xlim(left=0, right=x_max)
    
    # Format inset plot
    ax_inset.grid(True, alpha=0.3, linestyle=':')
    ax_inset.set_title('Zoomed View', fontsize=10)
    
    # Add connecting lines between main plot and inset to show relationship
    # Define the corners of the inset region in data coordinates
    inset_corners = [
        (0, inset_y_min),              # bottom left
        (x_max, inset_y_min),          # bottom right
        (x_max, inset_y_max),          # top right
        (0, inset_y_max)               # top left
    ]
    
    # Find the corresponding points in the main axes
    main_corners = [
        (0, inset_y_min),              # bottom left
        (x_max, inset_y_min),          # bottom right
        (x_max, inset_y_max),          # top right
        (0, inset_y_max)               # top left
    ]
    
    # Create connection patches between the inset and main axes
    for i in range(4):
        # Connect each corner to show the zoomed region
        con = ConnectionPatch(
            xyA=inset_corners[i], xyB=main_corners[i],
            coordsA="data", coordsB="data",
            axesA=ax_inset, axesB=ax_main,
            color="lightgray", linewidth=1, alpha=0.6
        )
        ax_inset.add_artist(con)
    
    # Format main axes with labels and title
    ax_main.set_xlabel('Iteration', fontsize=12)
    ax_main.set_ylabel('Episode Length', fontsize=12)
    ax_main.set_title(f'Comparison of Episode Length with Segment Analysis for both Environment Types', fontsize=14)
    
    # Use consistent tick marks for main axes
    # For x-axis, use ticks at every 100 iterations from 0 to x_max
    ax_main.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    
    # For y-axis, use ticks at every 100 units from 0 to y_max
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax_main.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    
    # Customize grid for main plot
    ax_main.grid(True, alpha=0.3, linestyle=':')
    ax_main.grid(which='minor', alpha=0.1, linestyle=':')
    
    # Add legend to main plot
    ax_main.legend(loc='lower right')
    
    # Apply tight layout to optimize spacing
    try:
        plt.tight_layout()
    except:
        print("Warning: Could not apply tight_layout. Figure may have suboptimal spacing.")
    
    # Save figure to specified path with high DPI
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved multi-run analysis plot with picture-in-picture to {output_path}")
    plt.close(fig)


# Example usage of the function and command-line interface
if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Create multi-run segment analysis plot from WandB exports and iteration data')
    
    # Required arguments
    parser.add_argument('csv_file', help='Path to WandB export CSV file')  # Single CSV file
    parser.add_argument('txt_file', help='Path to iteration data text file')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--output', '-o', default='multi_run_analysis.png', help='Output image path')
    parser.add_argument('--window', '-w', type=int, default=WINDOW_SIZE, help='Moving average window size')
    parser.add_argument('--segments', '-s', type=int, default=SEGMENT_COUNT, help='Number of segments for analysis')
    parser.add_argument('--run-names', '-n', nargs='+', help='Names for each run (e.g., "Run 1", "Run 2")')
    parser.add_argument('--run-descriptions', '-d', nargs='+', help='Descriptions for each run (e.g., "Terminal Right", "Terminal Left")')
    parser.add_argument('--exp-ids', '-e', nargs='+', help='Experiment IDs to extract from the CSV file')
    parser.add_argument('--vlines', '-v', type=int, nargs='+', help='Add custom vertical lines at specified x-coordinates')
    parser.add_argument('--xmax', '-x', type=int, default=1000, help='Maximum value for x-axis')
    parser.add_argument('--ymax', '-y', type=int, default=510, help='Maximum value for y-axis')
    parser.add_argument('--no-ma', action='store_true', help='Do not show moving average lines')
    parser.add_argument('--no-raw', action='store_true', help='Do not show raw data lines')
    parser.add_argument('--inset-min', type=float, default=450, help='Minimum value for inset y-axis')
    parser.add_argument('--inset-max', type=float, default=510, help='Maximum value for inset y-axis')
    parser.add_argument('--inset-width', type=float, default=0.5, help='Width of inset as fraction of figure size')
    parser.add_argument('--inset-height', type=float, default=0.5, help='Height of inset as fraction of figure size')
    
    args = parser.parse_args()
    
    # If we have multiple experiment IDs but just one CSV file, use the same CSV for all runs
    csv_files = [args.csv_file] * len(args.exp_ids) if args.exp_ids else [args.csv_file]
    
    # Call the main plotting function with parsed arguments
    create_multi_run_plot(
        csv_files=csv_files,
        txt_file=args.txt_file,
        output_path=args.output,
        run_names=args.run_names,
        run_descriptions=args.run_descriptions,
        exp_ids=args.exp_ids,
        window_size=args.window,
        num_segments=args.segments,
        custom_vlines=args.vlines,
        x_max=args.xmax,
        y_max=args.ymax,
        show_ma=not args.no_ma,
        show_raw=not args.no_raw,
        inset_y_min=args.inset_min,
        inset_y_max=args.inset_max,
        inset_size=(args.inset_width, args.inset_height)
    )