import os
import csv
import pandas as pd
from matplotlib import pyplot as plt

def save_to_csv(filename, data_dict, folder):
        """
        Appends a dictionary of data to a CSV file. 
        Creates the file with headers if it doesn't exist.
        """

        file_path = os.path.join(folder, filename)
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)


def get_agents_comparison_plot(agent_paths_dict, smoothing_window=1, output_folder="plots", prefix="train_comparison"):
    """
    Saves individual SVG plots for each variable found in the CSV logs.
    
    Args:
        agent_paths_dict: dict { "Agent Name": "path/to/stats.csv" }
        smoothing_window: int, size of rolling average window.
        output_folder: str, folder where SVG files will be saved.
    """

    # Load data
    agent_data = {}
    for name, path in agent_paths_dict.items():
        if os.path.exists(path):
            agent_data[name] = pd.read_csv(path)
        else:
            print(f"Warning: Path for {name} not found: {path}")

    if not agent_data:
        print("No data found to plot.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Identify Structure
    first_agent = list(agent_data.keys())[0]
    columns = list(agent_data[first_agent].columns)
    x_col = columns[0]
    metrics = columns[1:]
    
    clean_x_label = x_col.replace('_', ' ').title()

    # Loop through metrics and create individual plots
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        for agent_name, df in agent_data.items():
            if metric not in df.columns:
                continue
            y_values = df[metric]
            if smoothing_window > 1:
                y_values = y_values.rolling(window=smoothing_window, min_periods=1).mean()
            ax.plot(df[x_col], y_values, label=agent_name, linewidth=1.5)

        clean_y_label = metric.replace('_', ' ').title()
        ax.set_xlabel(clean_x_label)
        ax.set_ylabel(clean_y_label)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        file_name = f"{prefix}_{metric}.svg"
        save_path = os.path.join(output_folder, file_name)
        fig.tight_layout()
        fig.savefig(save_path, format='svg')
        plt.close(fig)
        
        print(f"Saved: {save_path}")
