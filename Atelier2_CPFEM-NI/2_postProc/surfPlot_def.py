## Import necessary libraries for interpolation and visualization
import matplotlib.pyplot as plt
import numpy as np

def read_header(file_path, num_lines=4):
    """Reads the header lines from the file."""
    header_lines = []
    with open(file_path, "r") as file:
        for _ in range(num_lines):
            header_lines.append(file.readline().strip())
    return header_lines

def load_afm_data(file_path, skiprows=4):
    """Loads AFM data from the file, skipping header lines."""
    try:
        return np.loadtxt(file_path, skiprows=skiprows)
    except ValueError as e:
        print(f"Error loading data: {e}")
        raise

def center_on_minimum(data_cropped, grid_x, grid_y):
    """Centers the cropped data on the minimum Z value with (0, 0) at the minimum."""
    # Find the index of the minimum Z value
    min_z_index = np.unravel_index(np.nanargmin(data_cropped), data_cropped.shape)
    
    # Get the coordinates of the minimum Z value
    min_x, min_y = grid_x[min_z_index], grid_y[min_z_index]
    
    # Shift the grid so that the minimum Z value is at (0, 0)
    grid_x_centered = grid_x - min_x
    grid_y_centered = grid_y - min_y
    
    return grid_x_centered, grid_y_centered

def visualize_data(data_cropped, grid_x_centered, grid_y_centered):
    """Visualizes the cropped and centered data."""
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(data_cropped, extent=(grid_x_centered.min(), grid_x_centered.max(), grid_y_centered.min(), grid_y_centered.max()), 
                         origin='lower', cmap='viridis', aspect='auto')
    plt.title("AFM Data Centered on Minimum Z Value")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.colorbar(heatmap, label="Height (nm)")
    plt.show()