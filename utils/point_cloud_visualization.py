import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, title='Point Cloud Visualization'):
    """
    Visualize point cloud using multiple methods
    
    Parameters:
    -----------
    point_cloud : np.ndarray or o3d.geometry.PointCloud
        Point cloud to visualize
    title : str, optional
        Title for the visualization
    """
    # Convert to numpy array if Open3D point cloud
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points)
    else:
        points = point_cloud

    # 1. Open3D Visualization
    def open3d_visualization():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Colorize point cloud (optional)
        if points.shape[1] > 3:
            colors = points[:, 3:]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualization options
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd)
        
        # Set up render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.7, 0.7, 0.7])  # Light gray background
        opt.point_size = 3.0  # Adjust point size
        
        vis.run()
        vis.destroy_window()

    # 2. Matplotlib 3D Scatter Plot
    def matplotlib_3d_scatter():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot
        scatter = ax.scatter(
            points[:, 0], 
            points[:, 1], 
            points[:, 2], 
            c=points[:, 2],  # Color by z-coordinate
            cmap='viridis', 
            alpha=0.7
        )
        
        # Add color bar
        plt.colorbar(scatter)
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()

    # 3. Matplotlib Subplots for Different Projections
    def matplotlib_projections():
        fig = plt.figure(figsize=(15, 5))
        
        # XY Plane
        ax1 = fig.add_subplot(131)
        ax1.scatter(points[:, 0], points[:, 1], alpha=0.5)
        ax1.set_title('XY Projection')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # XZ Plane
        ax2 = fig.add_subplot(132)
        ax2.scatter(points[:, 0], points[:, 2], alpha=0.5)
        ax2.set_title('XZ Projection')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        
        # YZ Plane
        ax3 = fig.add_subplot(133)
        ax3.scatter(points[:, 1], points[:, 2], alpha=0.5)
        ax3.set_title('YZ Projection')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        
        plt.tight_layout()
        plt.show()

    # Run visualizations
    open3d_visualization()
    # matplotlib_3d_scatter()   
    # matplotlib_projections()

def analyze_point_cloud(point_cloud):
    """
    Provide detailed analysis of the point cloud
    
    Parameters:
    -----------
    point_cloud : np.ndarray or o3d.geometry.PointCloud
        Point cloud to analyze
    
    Returns:
    --------
    dict
        Point cloud statistics
    """
    # Convert to numpy array
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points)
    else:
        points = point_cloud
    
    # Compute statistics
    stats = {
        'total_points': points.shape[0],
        'dimensions': points.shape[1],
        'mean': np.mean(points, axis=0),
        'std': np.std(points, axis=0),
        'min': np.min(points, axis=0),
        'max': np.max(points, axis=0),
        'bbox_size': np.max(points, axis=0) - np.min(points, axis=0)
    }
    
    # Print statistics
    print("Point Cloud Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return stats

# Example usage in your dataset context
def visualize_dataset_point_clouds(path):
    """
    Visualize point clouds from multiple samples in the dataset
    
    Parameters:
    -----------
    path : str
        Path to the dataset
    """
    import glob
    import os

    # sample_path = path
    
    # Find all sample directories
    # sample_dirs = [d for d in glob.glob(os.path.join(path, 'test_sample_*')) if os.path.isdir(d)]
    sample_dirs = [d for d in glob.glob(os.path.join(path, 'canonical_sample_*')) if os.path.isdir(d)]
    # sample_dirs = [d for d in glob.glob(os.path.join(path, 'sample_*')) if os.path.isdir(d)]
    
    # Visualize first few samples
    for sample_path in sample_dirs[:5]:  # Limit to first 5 samples
        # Read point cloud
        pcd_path = os.path.join(sample_path, 'pc.ply')
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        print(f"\nVisualizing point cloud from: {sample_path}")
        
        # Visualize
        visualize_point_cloud(pcd, title=f'Point Cloud - {os.path.basename(sample_path)}')
        
        # Analyze
        analyze_point_cloud(pcd)

def visualize_multiview_point_clouds(path):
    """
    Visualize point clouds from multiple samples in the dataset
    
    Parameters:
    -----------
    path : str
        Path to the dataset
    """
    import glob
    import os

    sample_path = path

    # Read point cloud
    pcd_path = os.path.join(sample_path, 'points3D_multipleview.ply')
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print(f"\nVisualizing point cloud from: {sample_path}")
    
    # Visualize
    visualize_point_cloud(pcd, title=f'Point Cloud - {os.path.basename(sample_path)}')
    
    # Analyze
    analyze_point_cloud(pcd)

def visualize_result_point_clouds(path):
    """
    Visualize point clouds from multiple samples in the dataset
    
    Parameters:
    -----------
    path : str
        Path to the dataset
    """
    import glob
    import os

    sample_path = path
    
    # Find all sample directories
    # sample_dirs = [d for d in glob.glob(os.path.join(path, 'canonical_sample_*')) if os.path.isdir(d)]
    # sample_dirs = [d for d in glob.glob(os.path.join(path, 'sample_*')) if os.path.isdir(d)]
    
    # Visualize first few samples
    # for sample_path in sample_dirs[:5]:  # Limit to first 5 samples
    # Read point cloud
    pcd_path = os.path.join(sample_path, 'point_cloud.ply')
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print(f"\nVisualizing point cloud from: {sample_path}")
    
    # Visualize
    visualize_point_cloud(pcd, title=f'Point Cloud - {os.path.basename(sample_path)}')
    
    # Analyze
    analyze_point_cloud(pcd)


# Call this in your script
visualize_result_point_clouds('/home/iulian/chole_ws/src/drrobot/output/p_short_exp_2/point_cloud/pose_conditioned_iteration_48000')
# visualize_dataset_point_clouds('/home/iulian/chole_ws/src/drrobot/data/prograsp_dataset_center_close_test')
# visualize_dataset_point_clouds('/home/iulian/chole_ws/src/drrobot/data/franka_emika_panda')
# visualize_result_point_clouds('/home/iulian/chole_ws/src/drrobot/output/franka_test/point_cloud/pose_conditioned_iteration_8000')