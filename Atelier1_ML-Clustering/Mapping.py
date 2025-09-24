import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture

def plot_pdf_with_deconvolution_on_axis(data, column, n_components, ax):
    # Plot histogram
    sns.histplot(data[column].dropna(), bins=30, kde=False, stat='density', color='lightgray', edgecolor='black', ax=ax)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data[[column]].dropna())
    
    # Generate x values for plotting the GMM components
    x = np.linspace(data[column].min(), data[column].max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    
    # Plot GMM components
    for i in range(n_components):
        ax.plot(x, responsibilities[:, i] * pdf, label=f'Component {i+1}')
    
    ax.plot(x, pdf, '-k', label='GMM Total')
    ax.set_title(f'PDF and GMM Deconvolution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.legend()

def plot_cdf_with_weibull_fit_on_axis(data, column, ax):
    # Sort data and compute CDF
    sorted_data = np.sort(data[column].dropna())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot empirical CDF
    ax.step(sorted_data, cdf, where='post', label='Empirical CDF', color='blue')
    
    # Fit Weibull distribution
    from scipy.stats import weibull_min
    params = weibull_min.fit(sorted_data, floc=0)
    x = np.linspace(sorted_data.min(), sorted_data.max(), 1000)
    weibull_cdf = weibull_min.cdf(x, *params)
    
    # Plot Weibull fit
    ax.plot(x, weibull_cdf, 'r-', label='Weibull Fit')
    
    ax.set_title(f'CDF and Weibull Fit of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('CDF')
    ax.legend() 

def plot_clustered_data(data, xdata, ydata, huedata, colors, result_dir, xDim=10, yDim=6, title='KMeans Clustering of Hardness and Modulus', xlabel='Hardness (GPa)', ylabel='Modulus (GPa)'):
    """
    Plot clustered data using the provided colors and save the plot.

    Parameters:
    - data: DataFrame
        The data containing 'HARDNESS_GPa', 'MODULUS_GPa', and 'Cluster' columns.
    - colors: list or dict
        The color palette for the clusters.
    - result_dir: Path or str
        The directory where the plot will be saved.
    - xDim: int, optional
        The width of the figure (default is 10).
    - yDim: int, optional
        The height of the figure (default is 6).
    """
    # Plot clustered data
    plt.figure(figsize=(xDim, yDim))
    sns.scatterplot(x=xdata, y=ydata, hue=huedata, palette=colors, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    file_path = result_dir / (title.replace(" ", "_") + '.png')
    plt.savefig(file_path)
    print(f"Clustered data plot saved to {file_path}")
    plt.show()

# Function to plot a map with given x, y, z data
def plot_map(x, y, z, title, xlabel, ylabel, ax, cmap='viridis', save_path=None):
    # Filter out non-finite values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    
    # Use the provided axis for plotting
    contour = ax.tricontourf(x, y, z, levels=14, cmap=cmap)
    cbar = plt.colorbar(contour, ax=ax)  # Add colorbar to the axis
    cbar.set_label(title)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Map plot saved to {save_path}")

# Function to create a grid and plot square pixels
def plot_pixel_map(x, y, z, title, xlabel, ylabel, xDim, yDim, cluster_colors, save_path=None):
    # Create a pivot table to structure data into a grid
    grid_data = pd.DataFrame({'X': x, 'Y': y, 'Cluster': z})
    pivot = grid_data.pivot(index='Y', columns='X', values='Cluster')
    
    # Sort the grid to ensure proper alignment
    pivot = pivot.sort_index(ascending=False)  # Y-axis should be inverted for plotting
    
    # Create a colormap for clusters
    unique_clusters = np.unique(z)
    cluster_cmap = [cluster_colors[cluster] for cluster in unique_clusters]
    cmap = plt.matplotlib.colors.ListedColormap(cluster_cmap)
    
    # Plot the grid as square pixels
    plt.figure(figsize=(xDim, yDim))
    plt.imshow(pivot, cmap=cmap, aspect='equal', interpolation='none')  # Square pixels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a legend for clusters
    handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=10, label=f'Cluster {cluster}')
               for cluster, color in zip(unique_clusters, cluster_cmap)]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save with tight layout to include the legend
        print(f"Pixel map plot saved to {save_path}")
    plt.show()