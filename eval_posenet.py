
import os
# os.environ['PATH'] = '/usr/local/cuda-11.6/bin:' + os.environ['PATH']
# if 'notebooks' not in os.listdir(os.getcwd()):
# os.chdir('../') #changing directories so that output/gsplat_full etc. exists

from contextlib import redirect_stdout
from video_api import initialize_gaussians
from gaussian_renderer import render
import sys 
import torch 
from PIL import Image
import numpy as np

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from utils.mujoco_utils import compute_camera_extrinsic_matrix, compute_camera_intrinsic_matrix
from utils.mujoco_utils import get_canonical_pose, set_xml_light_params, find_non_collision_pose, save_robot_metadata, ControlRobot
import mujoco
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, se3_to_SE3
from IPython.display import Image as IPImage
from moviepy import ImageSequenceClip
from scipy.spatial.transform import Rotation 
import cv2
from torchviz import make_dot
import pandas as pd
import random
from notebooks.utils import Utils
import argparse
from torchvision import transforms


def invert_H(H):
    R = H[:3, :3]
    t = H[:3, 3]
    R_inv = R.transpose()
    t_inv = - R_inv @ t
    H_inv = np.eye(4)
    H_inv[:3, :3] = R_inv
    H_inv[:3, 3] = t_inv
    return H_inv

def invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = - R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, [-1]] = t_inv
    return T_inv

# quaternion to rotation matrix
def quaternion_to_rotation_matrix(quaternion):

    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]
    ])
    return rotation_matrix

def estimate_transformation(pts_gt, pts_est):
    """
    Estimate the rigid transformation (R, t) that aligns pts_est to pts_gt.
    
    Args:
        pts_gt:  (N, 3) array of ground truth 3D points.
        pts_est: (N, 3) array of estimated 3D points.

    Returns:
        R (3x3 rotation matrix), t (3x1 translation vector)
    """
    # Compute centroids
    centroid_gt = np.mean(pts_gt, axis=0)
    centroid_est = np.mean(pts_est, axis=0)
    
    # Center the points
    P_prime = pts_gt - centroid_gt
    Q_prime = pts_est - centroid_est
    
    # Compute the covariance matrix
    H = Q_prime.T @ P_prime
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) should be 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_gt - R @ centroid_est
    
    return R, t


def main(args):

    ## ------ Initialize Gaussian Renderer ------ ##
    sys.argv = ['']
    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path=args.model_path, bg_color=[0, 0, 0])
    pose = gaussians.get_xyz
    w2e = torch.mean(gaussians._xyz, 0).clone().detach()

    T_wt_homogeneous = torch.cat([w2e, torch.tensor([1.0], device='cuda:0')])
    example_camera = sample_cameras[0]
    P = example_camera.projection_matrix.to('cuda:0')


    ## ------ Load PoseNet Model ------ ##
    from train_init_network import PoseNet, PoseNetTransformer
    if args.model_name.endswith("transformer"):
        model = PoseNetTransformer().to("cuda")
    else:
        model = PoseNet().to("cuda")
    # model_params= torch.load("./output/LND_pose/pose_net_chkpnt29000.pth")
    (model_params, trained_iter) = torch.load(f"./output/{args.model_name}/pose_net_chkpnt_{args.model_num}.pth")

    model.load_state_dict(model_params)

    model.eval()  # Set to evaluation mode


    for i in range(args.num_samples):

        ## ------ Display Reference Image ------ ##
        idx = random.randint(0, 10)
        img_idx = random.randint(1, 5)

        # data_dir = f"/home/iulian/chole_ws/src/sam2/data/surgripe/lnd_test"
        # original_image_path = data_dir + f"/{img_idx}.jpg" 
        # reference_image_path = data_dir + f"_segment_results//frame_{img_idx}_id_1.jpg" 
        data_dir = f"/home/iulian/chole_ws/src/drrobot/data/LND_pose_train/test_sample_{idx}"
        original_image_path = data_dir + f"/image_{img_idx}.jpg" 
        reference_image_path = original_image_path  
        # object_poses = np.load(os.path.join(data_dir, "object_poses.npy"))
        augment = transforms.Compose([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.GaussianBlur(kernel_size=3),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Only small translation and scale
                    # transforms.ToTensor(),
                ])
        ## read the image
        original_image = Image.open(original_image_path).convert('RGB')
        reference_image = Image.open(reference_image_path).convert('RGB')

        reference_image = torch.tensor(np.array(reference_image) / 255.0).permute(2, 0, 1).float().to("cuda")  # Normalized [0, 1]
        reference_image = Utils.pad_to_match_aspect(reference_image, 1.0)

        reference_image = F.interpolate(reference_image.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        reference_image_small = F.interpolate(reference_image.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        # Calculate target aspect ratio (reference image)
        target_aspect_ratio = reference_image.shape[2] / reference_image.shape[1]


        output = model(reference_image_small.unsqueeze(0))
        # print(output)

        ## turn the output to homogenous transformation matrix
        trans = output[0, :3]
        rot = output[0, 3:]
        rot = rot / torch.norm(rot)

        rot = rot.cpu().detach().numpy()

        ## turn quaternion to rotation matrix
        rot_matrix = Rotation.from_quat(rot).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = trans.cpu().detach().numpy()


        ## ------ Configure Camera Parameters ------ ##

        T_rotated_torch = torch.tensor(pose, dtype=example_camera.world_view_transform.dtype).transpose(0, 1).to(example_camera.data_device)

        example_camera.world_view_transform = T_rotated_torch


        ## ------ Render Init Gaussian ------ ##
        home_pose = [0.0, 0.0, 0.0]

        example_camera.joint_pose = torch.tensor(home_pose,).requires_grad_(True)
        example_camera.world_view_transform.requires_grad_(True)


        ## ------ Compare the final rendered image and the reference image with overlay ------ ##
        final_render = torch.clamp(render(example_camera, gaussians, background_color)['render'], 0, 1)
        padded_rendered_image = Utils.pad_to_match_aspect(final_render, target_aspect_ratio)

        # Convert images for visualization
        rendered_image_np = padded_rendered_image.detach().permute(1, 2, 0).cpu().numpy()
        ground_truth_image_np = reference_image.permute(1, 2, 0).cpu().numpy()

        # Alpha blending (overlay with transparency)
        alpha = 0.3  # Transparency factor
        blended_image = (alpha * rendered_image_np + (1 - alpha) * ground_truth_image_np)

        # Extract silhouette using threshold and contour detection
        rendered_mask = np.any(rendered_image_np > 0.05, axis=-1)  # Threshold for silhouette detection
        contours = measure.find_contours(rendered_mask.astype(float), 0.5)

        ## Plot the images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Ground Truth
        ax = axs[0]
        ax.set_title("Ground Truth Image")
        ax.imshow(augment(original_image))
        ax.axis('off')

        # Rendered Image
        ax = axs[1]
        ax.set_title("Rendered Image")
        ax.imshow(rendered_image_np)
        ax.axis('off')

        # Blended Visualization with Silhouette
        ax = axs[2]
        ax.set_title("Overlay with Silhouette")
        ax.imshow(blended_image)

        # Draw silhouette contours on the blended image
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        ax.axis('off')

        # plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='output/LND_short')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--model_num', type=int, default=20000)
    parser.add_argument('--model_name', type=str, default='LND_pose_train')

    args = parser.parse_args()

    main(args)