import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis
import seaborn as sns  
from matplotlib.colors import LogNorm  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_kam(data, output_folder, order, image_paths=None):
    def adjust_figure_size(image_paths):
        """Adjust figure size based on whether image paths are provided."""
        return (4, 3) if image_paths is None else (8, 6)

    def create_grids(data, col_E, col_H):
        """Create E and H grids from the data."""
        df_E = data.pivot_table(
            index="Y Position_µm", columns="X Position_µm", values=col_E, aggfunc="mean"
        )
        df_H = data.pivot_table(
            index="Y Position_µm", columns="X Position_µm", values=col_H, aggfunc="mean"
        )
        return df_E, df_H, df_E.values, df_H.values

    def get_neighbor_offsets(order):
        """Define neighbor offsets based on the order."""
        if order == 1:
            return [(0, 1), (0, -1), (1, 0), (-1, 0)]
        elif order == 2:
            return [(-2, 0), (2, 0), (0, -2), (0, 2),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            raise ValueError("order must be 1 or 2")

    def compute_kam_metric_diff(grid, df_reference, name, neighbor_offsets):
        """Compute KAM metric differences."""
        kam = np.full(grid.shape, np.nan, dtype=float)
        rows, cols = kam.shape

        for i in range(rows):
            for j in range(cols):
                if np.isnan(grid[i, j]):
                    continue
                center_val = grid[i, j]
                diffs = [
                    abs(grid[i + di, j + dj] - center_val)
                    for di, dj in neighbor_offsets
                    if 0 <= i + di < rows and 0 <= j + dj < cols and not np.isnan(grid[i + di, j + dj])
                ]
                if diffs:
                    kam[i, j] = np.mean(diffs)

        df_kam = pd.DataFrame(kam, index=df_reference.index, columns=df_reference.columns)
        df_kam.index.name = "Y Position_µm"
        df_kam.columns.name = "X Position_µm"

        return (
            df_kam
            .reset_index()
            .melt(id_vars="Y Position_µm", var_name="X Position_µm", value_name=name)
            .dropna(subset=[name])
        )

    def merge_flattened_data(data, df, metric_name):
        """Flatten and merge KAM data into the original dataframe."""
        pivot = df.pivot(index="Y Position_µm", columns="X Position_µm", values=metric_name)
        flattened = pivot.stack().reset_index()
        flattened.columns = ["Y Position_µm", "X Position_µm", metric_name]
        return data.merge(flattened, on=["X Position_µm", "Y Position_µm"], how="left")

    # --- 1. Data preparation ---
    print("Raw data columns:", data.columns.tolist())
    wid, hei = adjust_figure_size(image_paths)
    fig = plt.figure(figsize=(wid * 3, hei))

    col_E = "young's modulus value" if "young's modulus value" in data.columns else "MODULUS_GPa"
    col_H = "hardness value" if "hardness value" in data.columns else "HARDNESS_GPa"

    df_E, df_H, grid_E, grid_H = create_grids(data, col_E, col_H)
    neighbor_offsets = get_neighbor_offsets(order)

    # --- 2. Compute KAM metrics ---
    df_kamm = compute_kam_metric_diff(grid_E * grid_H, df_E, "KAMM", neighbor_offsets)
    df_kaem = compute_kam_metric_diff(grid_E, df_E, "KAEM", neighbor_offsets)
    df_kapm = compute_kam_metric_diff(grid_H, df_H, "KAPM", neighbor_offsets)
    df_kamm_ratio = compute_kam_metric_diff(grid_E / grid_H, df_E, "KAMM_RATIO", neighbor_offsets)

    grid_muE, grid_muH = np.nanmean(grid_E), np.nanmean(grid_H)
    df_kamm_normProd = compute_kam_metric_diff(
        (grid_E - grid_muE) * (grid_H - grid_muH), df_E, "KAMM_NORMPROD", neighbor_offsets
    )

    # --- 3. Merge results back into the original data ---
    data = merge_flattened_data(data, df_kamm, "KAMM")
    data = merge_flattened_data(data, df_kaem, "KAEM")
    data = merge_flattened_data(data, df_kapm, "KAPM")
    data = merge_flattened_data(data, df_kamm_ratio, "KAMM_RATIO")
    data = merge_flattened_data(data, df_kamm_normProd, "KAMM_NORMPROD")

    return data