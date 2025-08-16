#!/usr/bin/env python3
"""
Streamlit app for viewing completion logs from training runs.

Usage:
    streamlit run view_completions.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Page config
st.set_page_config(
    page_title="Completion Logs Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better formatting
st.markdown("""
<style>
    .stDataFrame {
        font-size: 14px;
    }
    .prompt-box, .completion-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    .reward-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .reward-negative {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of entries."""
    entries = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return entries


def find_completion_logs_dir(base_path: str) -> Optional[Path]:
    """Find completion_logs directory within the given path."""
    base = Path(base_path)
    
    # Check if the path itself is completion_logs
    if base.name == "completion_logs" and base.exists():
        return base
    
    # Check for completion_logs subdirectory
    completion_logs = base / "completion_logs"
    if completion_logs.exists():
        return completion_logs
    
    # Search recursively (limited depth)
    for subdir in base.rglob("completion_logs"):
        return subdir  # Return first found
    
    return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def find_all_runs_with_logs(base_path: str, max_depth: int = 4) -> List[str]:
    """Find all directories containing completion_logs within base_path."""
    base = Path(base_path)
    if not base.exists():
        return []
    
    runs = []
    
    # Use rglob with pattern to find all completion_logs directories
    for logs_dir in base.rglob("completion_logs"):
        # Get the parent directory (the run directory)
        run_dir = logs_dir.parent
        # Get relative path from base for cleaner display
        try:
            rel_path = run_dir.relative_to(base)
            runs.append(str(rel_path))
        except ValueError:
            # If not relative to base, use absolute path
            runs.append(str(run_dir))
    
    # Sort for consistent ordering
    runs.sort()
    return runs


def format_prompt(prompt: Any) -> str:
    """Format prompt for display."""
    if isinstance(prompt, list):
        # Handle message format with clear separation
        formatted = []
        for i, msg in enumerate(prompt):
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                # Add visual separation with lines and spacing
                if i > 0:
                    formatted.append("\n" + "─" * 10 + "\n")
                formatted.append(f"【{role}】\n\n{content}")
        return '\n'.join(formatted)
    return str(prompt)


def format_completion(completion: Any) -> str:
    """Format completion for display."""
    if isinstance(completion, list) and completion:
        if isinstance(completion[0], dict):
            return completion[0].get('content', str(completion[0]))
        return str(completion[0])
    return str(completion)


def main():
    st.title("🔍 Completion Logs Viewer")
    st.markdown("Inspect completion logs from your training runs")
    
    # Sidebar for directory selection
    with st.sidebar:
        st.header("📁 Run Selection")
        
        # Base folder input
        default_base = st.session_state.get('base_folder', 'artifacts')
        base_folder = st.text_input(
            "Base folder:",
            value=default_base,
            placeholder="e.g., artifacts",
            help="Base folder to search for runs with completion logs"
        )
        
        # Check if base folder changed
        if base_folder != st.session_state.get('base_folder', ''):
            st.session_state['base_folder'] = base_folder
            # Clear selected run when base folder changes
            if 'selected_run' in st.session_state:
                del st.session_state['selected_run']
            # Clear cache to force rescan
            st.cache_data.clear()
        
        # Find all runs with completion logs
        if base_folder and Path(base_folder).exists():
            # Only show spinner on first load or after refresh
            if 'available_runs' not in st.session_state or st.session_state.get('last_base_folder') != base_folder:
                with st.spinner("Scanning for runs..."):
                    available_runs = find_all_runs_with_logs(base_folder)
                st.session_state['last_base_folder'] = base_folder
            else:
                available_runs = find_all_runs_with_logs(base_folder)
            
            if available_runs:
                st.success(f"Found {len(available_runs)} runs with logs")
                
                # Run selector dropdown
                run_options = ["-- Select a run --"] + available_runs
                selected_run = st.selectbox(
                    "Select a run:",
                    options=run_options,
                    index=run_options.index(st.session_state.get('selected_run')) 
                        if st.session_state.get('selected_run') in run_options else 0,
                    help="Choose a run to inspect its completion logs"
                )
                
                st.session_state['selected_run'] = selected_run
                
                # Only proceed if a valid run is selected
                if selected_run != "-- Select a run --":
                    # Build full path to the selected run
                    dir_path = str(Path(base_folder) / selected_run)
                    
                    # Find completion logs directory
                    logs_dir = find_completion_logs_dir(dir_path)
                    
                    if logs_dir:
                        # List available log files
                        jsonl_files = list(logs_dir.glob("*.jsonl"))
                        
                        if jsonl_files:
                            file_options = ["-- Select a log file --"] + [f.name for f in jsonl_files]
                            selected_file = st.selectbox(
                                "Select log file:",
                                file_options,
                                index=0
                            )
                        
                            if selected_file and selected_file != "-- Select a log file --":
                                selected_path = logs_dir / selected_file
                                
                                # Load the data
                                with st.spinner("Loading logs..."):
                                    entries = load_jsonl_file(selected_path)
                                
                                if entries:
                                    st.success(f"Loaded {len(entries)} entries")
                                    
                                    # Filtering options
                                    st.header("🎯 Filters")
                                    
                                    # Reward range filter
                                    rewards = [e.get('reward', 0) for e in entries]
                                    min_reward, max_reward = min(rewards), max(rewards)
                                    
                                    if min_reward == max_reward:
                                        # All rewards are the same - show info instead of slider
                                        st.info(f"All rewards have the same value: {min_reward:.4f}")
                                        reward_range = (min_reward, max_reward)
                                    else:
                                        reward_range = st.slider(
                                            "Reward range:",
                                            min_value=float(min_reward),
                                            max_value=float(max_reward),
                                            value=(float(min_reward), float(max_reward)),
                                            step=0.01
                                        )
                                    
                                    # Step range filter (only show if steps exist)
                                    steps = [e.get('step') for e in entries if e.get('step') is not None]
                                    if steps and len(set(steps)) > 1:  # Only show slider if there are different step values
                                        min_step, max_step = min(steps), max(steps)
                                        step_range = st.slider(
                                            "Step range:",
                                            min_value=min_step,
                                            max_value=max_step,
                                            value=(min_step, max_step)
                                        )
                                    elif steps and len(set(steps)) == 1:
                                        # All entries have the same step value
                                        st.info(f"All entries are from step {steps[0]}")
                                        step_range = None
                                    else:
                                        step_range = None
                                else:
                                    st.warning("No entries found in file")
                                    entries = []
                            else:
                                st.info("Please select a log file")
                                entries = []
                        else:
                            st.warning("No JSONL files found in completion_logs")
                            entries = []
                    else:
                        st.error(f"Could not find completion_logs directory")
                        entries = []
                else:
                    st.info("Please select a run to view its logs")
                    entries = []
            else:
                st.warning("No runs with completion logs found in this folder")
                entries = []
        elif base_folder:
            st.error(f"Base folder '{base_folder}' does not exist")
            entries = []
        else:
            st.info("Enter a base folder to search for runs")
            entries = []
    
    # Main content area
    if entries:
        # Apply filters
        filtered_entries = entries.copy()
        
        # Filter by reward range
        filtered_entries = [
            e for e in filtered_entries 
            if reward_range[0] <= e.get('reward', 0) <= reward_range[1]
        ]
        
        # Filter by step range (only if steps exist)
        if step_range is not None:
            filtered_entries = [
                e for e in filtered_entries 
                if step_range[0] <= e.get('step', 0) <= step_range[1]
            ]
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Statistics & Charts", "📋 Full Table"])
        
        with tab1:
            # Headline Statistics
            rewards_filtered = [e.get('reward', 0) for e in filtered_entries]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Entries", len(filtered_entries))
            with col2:
                if rewards_filtered:
                    st.metric("Mean", f"{sum(rewards_filtered)/len(rewards_filtered):.4f}")
                else:
                    st.metric("Mean", "N/A")
            with col3:
                if rewards_filtered:
                    import numpy as np
                    st.metric("Std Dev", f"{np.std(rewards_filtered):.4f}")
                else:
                    st.metric("Std Dev", "N/A")
            with col4:
                if rewards_filtered:
                    st.metric("Min", f"{min(rewards_filtered):.4f}")
                else:
                    st.metric("Min", "N/A")
            with col5:
                if rewards_filtered:
                    st.metric("Max", f"{max(rewards_filtered):.4f}")
                else:
                    st.metric("Max", "N/A")
        
            
            # Charts section
            st.markdown("---")
            if filtered_entries:
                df = pd.DataFrame(filtered_entries)
                
                if 'reward' in df.columns:
                    # Simple histogram with clear bins
                    st.subheader("Reward Distribution")
                    
                    import numpy as np
                    
                    # Get reward data
                    rewards_array = df['reward'].values
                    min_reward = rewards_array.min()
                    max_reward = rewards_array.max()
                    
                    # Handle edge cases
                    if min_reward == max_reward:
                        # All rewards are the same
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Bar(
                            x=[f"{min_reward:.3f}"],
                            y=[len(rewards_array)],
                            marker_color='#4CAF50',
                            width=0.5
                        ))
                        fig_hist.update_layout(
                            xaxis_title="Reward",
                            yaxis_title="Number of Rollouts",
                            showlegend=False,
                            height=400,
                            template="plotly_white"
                        )
                    else:
                        # Create evenly spaced bins
                        num_bins = min(20, len(rewards_array))  # Don't have more bins than data points
                        
                        # Round to nice numbers for bin edges
                        range_val = max_reward - min_reward
                        if range_val < 0.01:
                            # Very small range, use more precision
                            bin_size = range_val / num_bins
                        else:
                            # Round bin size to a nice number
                            bin_size = range_val / num_bins
                            # Round to 3 significant figures
                            magnitude = 10 ** np.floor(np.log10(bin_size))
                            bin_size = np.round(bin_size / magnitude) * magnitude
                        
                        # Create bin edges
                        bins = np.arange(min_reward, max_reward + bin_size, bin_size)
                        
                        # Calculate histogram manually for clearer control
                        hist, bin_edges = np.histogram(rewards_array, bins=bins)
                        
                        # Create bar chart with explicit bins
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        bin_labels = [f"{bin_edges[i]:.3f} to {bin_edges[i+1]:.3f}" 
                                     for i in range(len(bin_edges)-1)]
                        
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Bar(
                            x=bin_centers,
                            y=hist,
                            width=bin_size * 0.9,  # Slight gap between bars
                            marker_color='#4CAF50',
                            hovertemplate='Reward Range: %{text}<br>Count: %{y}<extra></extra>',
                            text=bin_labels
                        ))
                        
                        fig_hist.update_layout(
                            xaxis_title="Reward",
                            yaxis_title="Number of Rollouts",
                            showlegend=False,
                            height=400,
                            template="plotly_white",
                            xaxis=dict(
                                tickmode='linear',
                                tick0=min_reward,
                                dtick=bin_size * 2  # Show every other tick for clarity
                            )
                        )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Check if this is a training run (has steps) or eval run
                    has_steps = 'step' in df.columns and df['step'].notna().any()
                    
                    if has_steps:
                        # Training run - show reward over steps
                        st.subheader("Reward over Training Steps")
                        
                        # Group by step and calculate mean
                        step_rewards = df.groupby('step')['reward'].mean().reset_index()
                        
                        fig_steps = go.Figure()
                        
                        # Add mean rewards as markers only
                        fig_steps.add_trace(go.Scatter(
                            x=step_rewards['step'],
                            y=step_rewards['reward'],
                            mode='markers',
                            name='Mean Reward',
                            marker=dict(size=8, color='blue'),
                            hovertemplate='Step: %{x}<br>Mean Reward: %{y:.4f}<extra></extra>'
                        ))
                        
                        fig_steps.update_layout(
                            xaxis_title="Training Step",
                            yaxis_title="Mean Reward",
                            showlegend=False,
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig_steps, use_container_width=True)
                    else:
                        # Eval run - show individual rewards by index
                        st.subheader("Individual Rewards (Eval Run)")
                        df['index'] = range(len(df))
                        
                        fig_scatter = go.Figure()
                        
                        # Add scatter plot with entry index
                        fig_scatter.add_trace(go.Scatter(
                            x=df['index'],
                            y=df['reward'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=df['reward'],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="Reward")
                            ),
                            hovertemplate='Entry: %{x}<br>Reward: %{y:.4f}<extra></extra>'
                        ))
                        
                        # Add mean line for reference
                        mean_reward = df['reward'].mean()
                        fig_scatter.add_hline(
                            y=mean_reward,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Mean: {mean_reward:.3f}",
                            annotation_position="right"
                        )
                        
                        fig_scatter.update_layout(
                            xaxis_title="Entry Index",
                            yaxis_title="Reward",
                            showlegend=False,
                            height=400,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            # Full table view with prompts and completions
            st.subheader(f"Showing {len(filtered_entries)} entries")
            
            # Create comprehensive DataFrame for display
            table_data = []
            for entry in filtered_entries:
                row = {
                    'Step': entry.get('step', ''),
                    'Task ID': entry.get('task_id', ''),
                    'Prompt': format_prompt(entry.get('prompt', '')),
                    'Completion': format_completion(entry.get('completion', '')),
                    'Reward': entry.get('reward', 0),
                }
                
                # Add auxiliary info columns if present
                aux_info = entry.get('aux_info', {})
                if aux_info:
                    # Add common aux fields as separate columns
                    if 'segment' in aux_info:
                        row['Segment'] = aux_info['segment']
                    if 'accuracy' in aux_info:
                        row['Accuracy'] = aux_info['accuracy']
                    if 'inverted' in aux_info:
                        row['Inverted'] = aux_info['inverted']
                    if 'raw_score' in aux_info:
                        row['Raw Score'] = aux_info['raw_score']
                    if 'criterion' in aux_info:
                        row['Criterion'] = aux_info['criterion']  # Show full criterion
                    if 'reasoning' in aux_info:
                        row['Reasoning'] = aux_info['reasoning']  # Show full reasoning
                    if 'error' in aux_info:
                        row['Error'] = aux_info['error']
                
                table_data.append(row)
            
            df_display = pd.DataFrame(table_data)
            
            # Configure column display settings
            column_config = {
                'Step': st.column_config.NumberColumn(
                    'Step',
                    width='small',
                ),
                'Task ID': st.column_config.TextColumn(
                    'Task ID',
                    width='small',
                ),
                'Prompt': st.column_config.TextColumn(
                    'Prompt',
                    width='large',
                    help='Full prompt text'
                ),
                'Completion': st.column_config.TextColumn(
                    'Completion',
                    width='large',
                    help='Model completion'
                ),
                'Reward': st.column_config.NumberColumn(
                    'Reward',
                    format='%.4f',
                    width='small',
                ),
            }
            
            # Add config for aux columns if they exist
            for col in df_display.columns:
                if col == 'Accuracy' or col == 'Raw Score':
                    column_config[col] = st.column_config.NumberColumn(
                        col,
                        format='%.4f',
                        width='small'
                    )
                elif col == 'Inverted':
                    column_config[col] = st.column_config.CheckboxColumn(
                        col,
                        width='small'
                    )
            
            # Get min and max for scaling colors
            reward_min = df_display['Reward'].min()
            reward_max = df_display['Reward'].max()
            
            # Apply continuous color gradient to reward column
            def color_reward(val):
                """Color rewards with continuous gradient from red to green."""
                if pd.isna(val):
                    return ''
                
                # Normalize value to 0-1 range
                if reward_max == reward_min:
                    # All same value
                    norm = 0.5
                else:
                    norm = (val - reward_min) / (reward_max - reward_min)
                
                # Create color gradient from red (0) to yellow (0.5) to green (1)
                if norm < 0.5:
                    # Red to yellow
                    red = 255
                    green = int(255 * (norm * 2))
                    blue = 0
                else:
                    # Yellow to green
                    red = int(255 * (2 - norm * 2))
                    green = 255
                    blue = 0
                
                return f'background-color: rgba({red}, {green}, {blue}, 0.3)'
            
            # Apply styling to the dataframe
            styled_df = df_display.style.applymap(
                color_reward,
                subset=['Reward']
            )
            
            # Display the styled dataframe
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                height=800,  # Large height for better viewing
            )
    
    else:
        if base_folder and Path(base_folder).exists():
            st.info("No data to display. Please check your run selection and filters.")
        else:
            # Welcome message
            st.markdown("""
            ### Welcome to the Completion Logs Viewer! 
            
            To get started:
            1. Enter a base folder in the sidebar (default: `artifacts`)
            2. Select a run from the dropdown list of available runs
            3. Choose a log file if multiple are available
            4. Use filters to narrow down the entries
            5. Explore the data through the Statistics & Charts and Full Table tabs
            
            The viewer will automatically scan for all runs with `completion_logs` folders.
            """)


if __name__ == "__main__":
    main()