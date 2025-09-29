import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, kurtosis
import seaborn as sns  
from matplotlib.colors import LogNorm  

def compute_kam(data, output_folder, order, image_paths=None):
    # --- 1. Data preparation ---
    data_copy = data.copy()
    print("Raw data columns:", data.columns.tolist())

    # Ensure we have 'MODULUS_GPa' and 'HARDNESS_GPa' columns
    if "MODULUS_GPa" in data_copy.columns:
        data_copy.rename(columns={"MODULUS_GPa": "modulus"}, inplace=True)  
    if "HARDNESS_GPa" in data_copy.columns:
        data_copy.rename(columns={"HARDNESS_GPa": "hardness"}, inplace=True)
    print("Columns after renaming:", data_copy.columns.tolist())

    # Adjust figure size based on whether image_paths is provided
    if image_paths is None:
        wid, hei = 4, 3
    else:
        wid, hei = 8, 6

    # Create a single figure for 3 subplots side by side ( heatmaps of KAMM, KAEM, KAPM)
    fig = plt.figure(figsize=(wid * 3, hei))

    # --- 2. Create E and H grids ---
    data_filtered = data_copy.copy()
    col_E = "young's modulus value" if "young's modulus value" in data_filtered.columns else "modulus"
    col_H = "hardness value"       if "hardness value"       in data_filtered.columns else "hardness"

    df_E = data_filtered.pivot_table(
        index="Y Position_µm", columns="X Position_µm", values=col_E, aggfunc="mean"
    )
    df_H = data_filtered.pivot_table(
        index="Y Position_µm", columns="X Position_µm", values=col_H, aggfunc="mean"
    )
    grid_E = df_E.values
    grid_H = df_H.values

    # Define neighbor offsets based on order
    if order == 1:
        neighbor_offsets = [(0,1), (0,-1), (1,0), (-1,0)]
    elif order == 2:
        neighbor_offsets = [(-2,0), (2,0), (0,-2), (0,2),
                            (-1,-1), (-1,1), (1,-1), (1,1)]
    else:
        raise ValueError("order must be 1 or 2")

    def compute_kam_metric_diff(grid, df_reference, name):
        
        # Initialize KAM array
        kam = np.full(grid.shape, np.nan, dtype=float)
        rows, cols = kam.shape

        # Loop over each cell to compute neighbor differences
        for i in range(rows):
            for j in range(cols):
                if np.isnan(grid[i, j]):
                    continue
                center_val = grid[i, j]
                diffs = []
                for di, dj in neighbor_offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(grid[ni, nj]):
                        diffs.append(abs(grid[ni, nj] - center_val))
                if diffs:
                    kam[i, j] = np.mean(diffs)

        # Build DataFrame for output
        df_kam = pd.DataFrame(kam, index=df_reference.index, columns=df_reference.columns)
        df_kam.index.name = "Y Position_µm"
        df_kam.columns.name = "X Position_µm"

        # Melt to long format and merge with modulus/hardness values
        df_kam_long = (
            df_kam
            .reset_index()
            .melt(id_vars="Y Position_µm", var_name="X Position_µm", value_name=name)
            .dropna(subset=[name])
        )
        df_mod   = data_filtered.pivot_table(index="Y Position_µm", columns="X Position_µm", values="modulus",   aggfunc="mean")
        df_hard  = data_filtered.pivot_table(index="Y Position_µm", columns="X Position_µm", values="hardness", aggfunc="mean")
        df_mod_l = df_mod.reset_index().melt(id_vars="Y Position_µm", var_name="X Position_µm", value_name="modulus")
        df_hard_l= df_hard.reset_index().melt(id_vars="Y Position_µm", var_name="X Position_µm", value_name="hardness")

        df_kam_long = (
            df_kam_long
            .merge(df_mod_l,  on=["Y Position_µm","X Position_µm"], how="left")
            .merge(df_hard_l, on=["Y Position_µm","X Position_µm"], how="left")
        )

        # Save results CSV
        csv_path = os.path.join(output_folder, f"{name}_results.csv")
        df_kam_long.to_csv(csv_path, index=False)
        print(f"{name} results saved to {csv_path}")

        return df_kam_long
    
    # Compute three variants: combined (KAMM), E-only (KAEM), H-only (KAPM)
    # Direct multiplication: P₁ × P₂ gives an interaction field (highlighting where both are strong).
    df_kamm = compute_kam_metric_diff(grid_E * grid_H, df_E, "KAMM")
    df_kaem = compute_kam_metric_diff(grid_E,          df_E, "KAEM")
    df_kapm = compute_kam_metric_diff(grid_H,          df_H, "KAPM")
    # Ratio: P₁/P₂ (or log-ratio) for contrast.
    df_kamm_ratio = compute_kam_metric_diff(grid_E / grid_H, df_E, "KAMM_RATIO")
    # Normalized product: (P₁ – μ₁)(P₂ – μ₂) to highlight covariance-like behavior.
    grid_muE = np.nanmean(grid_E)
    grid_muH = np.nanmean(grid_H)
    df_kamm_normProd = compute_kam_metric_diff((grid_E - grid_muE) * (grid_H - grid_muH), df_E, "KAMM_NORMPROD")

    # Adjust layout and show side-by-side heatmaps
    plt.tight_layout()
    plt.show()
 
    return df_kamm, df_kaem, df_kapm, df_kamm_ratio, df_kamm_normProd