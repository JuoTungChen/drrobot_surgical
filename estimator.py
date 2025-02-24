
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
from pytorch_msssim import ssim


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


class PoseEstimator:

    def __init__(self, model_name, model_num):
        self.root_dir = "/home/iulian/chole_ws/src/drrobot"
        ## ------ Initialize Gaussian Renderer ------ ##
        sys.argv = ['']
        self.gaussians, self.background_color, sample_cameras, self.kinematic_chain = initialize_gaussians(model_path=self.root_dir+"/output/LND_short", bg_color=[0, 0, 0])
        self.pc = self.gaussians._xyz
        self.camera = sample_cameras[0]
        self.tooltip_l_idx, self.tooltip_r_idx = self.calculate_tooltip()
        self.pitch_screw_idx = self.calculate_pitch_screw_point()

        # Initialize parameters
        # self.home_pose = [0.0, 0.0, 0.0]
        self.home_pose = [0.0, 0.2, -0.5]


        # camera parameters
        self.P = self.camera.projection_matrix
        self.image_width = 256
        self.image_height = 256
        self.original_image_width = 1920
        self.original_image_height = 1080

        self.joint_pose_result = torch.tensor(self.home_pose, requires_grad=True, device="cuda")  # Initial joint pose
        self.world_view_transform_result = self.camera.world_view_transform.clone().detach().to("cuda").requires_grad_(True) # Initial camera transform

        self.rotation_result = self.camera.world_view_transform[:3, :3].clone().detach().to("cuda").requires_grad_(True)
        self.translation_result = self.camera.world_view_transform[3, :3].clone().detach().to("cuda").requires_grad_(True)

        ## ------ Setting Optimization Parameters ------ ##

        self.joint_pose_lr = 5e-4
        self.rotation_lr = 0.01
        # translation_lr = 1e-4
        self.translation_lr = 3e-4

        # Define optimizer with separate learning rates
        # self.optimizer = torch.optim.Adam([
        #     {"params": self.joint_pose_result, "lr": self.joint_pose_lr},
        #     {"params": self.world_view_transform_result, "lr": self.translation_lr},
        # ])



        self.num_iterations = 1000

        # self.loss_fn = torch.nn.MSELoss()
        def combined_loss(rendered, reference, alpha=0.8):
            mse = F.mse_loss(rendered, reference)
            
            # Ensure inputs are 4D: (batch, channels, height, width)
            rendered = rendered.unsqueeze(0) if rendered.dim() == 3 else rendered
            reference = reference.unsqueeze(0) if reference.dim() == 3 else reference
            
            ssim_l = 1 - ssim(rendered, reference, data_range=1.0)
            return alpha * ssim_l + (1 - alpha) * mse
        
        self.loss_fn = combined_loss

        self.video = []


        ## ------ Load PoseNet Model ------ ##
        from train_init_network import PoseNet, PoseNetTransformer

        if model_name.endswith("transformer"):
            self.model = PoseNetTransformer().to("cuda")
        else:
            self.model = PoseNet().to("cuda")
            
        # model_params= torch.load("./output/LND_pose/pose_net_chkpnt29000.pth")
        (model_params, trained_iter) = torch.load(self.root_dir + f"/output/{model_name}/pose_net_chkpnt_{model_num}.pth")

        self.model.load_state_dict(model_params)

        self.model.eval()  # Set to evaluation mode
        self.pose = np.eye(4)
    

    def calculate_tooltip(self, x_threshold=-0.003, z_threshold=-0.005):
        """
        calculate for the points in the gaussians corresponding to the tool tip and middle point
        """
        min_z, min_z_idx = torch.min(self.gaussians._xyz[:, 2], 0)
        # ## find the point with abs value of x < 0.0005 and z < -0.5 with min y value and its index
        filtered_indices = torch.logical_and(self.gaussians._xyz[:, 0] < x_threshold, self.gaussians._xyz[:, 2] < z_threshold)
        filtered_points = self.gaussians._xyz[filtered_indices]
        min_y, min_y_idx = torch.min(filtered_points[:, 1], 0)
        original_min_y_idx = torch.nonzero(filtered_indices)[min_y_idx][0]
        # middle_point = (self.gaussians._xyz[min_z_idx] + self.gaussians._xyz[original_min_y_idx]) / 2

        return min_z_idx, original_min_y_idx

    def calculate_pitch_screw_point(self, x_threshold=0.003, z_offset=-0.005, z_threshold=0.001):

        ## find the point with x and y around 0 and z = 0.005
        filtered_indices = torch.logical_and(torch.logical_and(torch.abs(self.gaussians._xyz[:, 0]) < x_threshold, torch.abs(self.gaussians._xyz[:, 1]) < x_threshold), torch.abs(self.gaussians._xyz[:, 2] + z_offset) < 0.001)
        filtered_points = self.gaussians._xyz[filtered_indices]
        # print(filtered_points)
        min_x, min_x_idx = torch.min(torch.abs(filtered_points[:, 0]), 0)
        point_idx = torch.nonzero(filtered_indices)[min_x_idx][0]

        return point_idx

    def image_reader(self, img_path):
        '''
        read image and preprocessing
        '''

        ## ------ Display Reference Image ------ ##
        ## read the image
        # original_image = Image.open(original_image_path).convert('RGB')
        self.reference_image = Image.open(img_path).convert('RGB')

        self.reference_image = torch.tensor(np.array(self.reference_image) / 255.0).permute(2, 0, 1).float().to("cuda")  # Normalized [0, 1]
        self.reference_image = Utils.pad_to_match_aspect(self.reference_image, 1.0)

        self.reference_image = F.interpolate(self.reference_image.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        self.reference_image_small = F.interpolate(self.reference_image.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        # Calculate target aspect ratio (reference image)
        self.target_aspect_ratio = self.reference_image.shape[2] / self.reference_image.shape[1]

        return self.reference_image_small
    

    def convert_transformation(self):

        ## calculate middle point
        middle_point = (self.pc[self.tooltip_l_idx] + self.pc[self.tooltip_r_idx]) / 2
        screw_point = self.pc[self.pitch_screw_idx]

        theta = self.joint_pose_result[0].item() + np.pi/2
        theta_z = np.pi
        R_z = torch.tensor([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z),  np.cos(theta_z), 0],
            [0,              0,             1]
        ], device='cuda:0')

        R_x = torch.tensor([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ], device='cuda:0')
        
        ori_rotation = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device='cuda:0')

        middle_point_homogeneous = torch.cat([torch.tensor(screw_point.clone().detach(), device='cuda:0'), torch.tensor([1.0], device='cuda:0')])
        T_wt_homogeneous = torch.eye(4, device='cuda:0')
        T_wt_homogeneous[:3, :3] = R_z @ R_x
        T_wt_homogeneous[:3, 3] = middle_point_homogeneous[:3]
        T_wc = self.world_view_transform_result  #.transpose(1, 0)

        # Calculate the position of the surgical tool in the camera frame
        T_tc_homogeneous = T_wt_homogeneous.T @ T_wc

        projected_pose = self.P.T.detach().cpu() @ T_tc_homogeneous.detach().cpu()

        # Normalize by the homogeneous coordinate (w) to get 2D image coordinates
        x_2d = ((projected_pose[0] / projected_pose[2] + 1) * 0.5) * self.image_width
        y_2d = ((projected_pose[1] / projected_pose[2] + 1) * 0.5) * self.image_height

        tool_pose_2d = (x_2d, y_2d)
        T_tc = T_tc_homogeneous.detach().cpu().numpy().transpose(1, 0)

        return tool_pose_2d, T_tc

    # def convert_to_shell_transform(self):
        


    def display_render(self):
        ## ------ Compare the final rendered image and the reference image with overlay ------ ##
        final_render = torch.clamp(render(self.camera, self.gaussians, self.background_color)['render'], 0, 1)
        padded_rendered_image = Utils.pad_to_match_aspect(final_render, self.target_aspect_ratio)

        # Convert images for visualization
        rendered_image_np = padded_rendered_image.detach().permute(1, 2, 0).cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(rendered_image_np)
        plt.axis('off')
        plt.show()


    def display_overlay(self):
        ## ------ Compare the final rendered image and the reference image with overlay ------ ##
        # final_render = torch.clamp(render(self.camera, self.gaussians, self.background_color)['render'], 0, 1)
        # padded_rendered_image = Utils.pad_to_match_aspect(final_render, self.target_aspect_ratio)

        # Convert images for visualization
        rendered_image_np = self.video[-1].detach().permute(1, 2, 0).cpu().numpy()
        # rendered_image_np = padded_rendered_image.detach().permute(1, 2, 0).cpu().numpy()
        ground_truth_image_np = self.reference_image.permute(1, 2, 0).cpu().numpy()

        # Alpha blending (overlay with transparency)
        alpha = 0.3
        blended_image = (alpha * rendered_image_np + (1 - alpha) * ground_truth_image_np)

        # Extract silhouette using threshold and contour detection
        rendered_mask = np.any(rendered_image_np > 0.05, axis=-1)  # Threshold for silhouette detection
        contours = measure.find_contours(rendered_mask.astype(float), 0.5)

        ## Plot the images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Ground Truth
        ax = axs[0]
        ax.set_title("Ground Truth Image")
        ax.imshow(ground_truth_image_np)
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

        plt.show()

    def get_preset_pose(self):

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = 0.1
        cam.azimuth = -90.0
        cam.elevation = -10
        cam.lookat = (0,0,-0.0)
        w2v = compute_camera_extrinsic_matrix(cam)

        theta = np.pi - np.pi/8  # 90 degrees in radians
        R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        R = w2v[:3, :3]
        t = w2v[:3, 3]
        t[0] += 0.01
        t[1] += 0.01
        # Apply the rotation
        R_rotated = R_z @ R  # Rotate the original rotation matrix
        t_rotated = t  # Rotate the translation vector

        # Combine back into a transformation matrix
        T_rotated = np.eye(4)
        T_rotated[:3, :3] = R_rotated
        T_rotated[:3, 3] = t_rotated

        print("get preset pose\n", T_rotated)


        return T_rotated

    def get_posenet_prediction(self, img):

        ## ------ PoseNet Prediction ------ ##
        output = self.model(img.unsqueeze(0))

        pose = np.eye(4)
        # turn the output to homogenous transformation matrix
        trans = output[0, :3]
        rot = output[0, 3:]
        rot = rot / torch.norm(rot)
        rot = rot.cpu().detach().numpy()

        # turn quaternion to rotation matrix
        rot_matrix = Rotation.from_quat(rot).as_matrix()

        pose[:3, :3] = rot_matrix
        pose[:3, 3] = trans.cpu().detach().numpy()

        return pose


    def inhibit_x_rotation(self, euler):
        """
        Inhibit the rotation around the x-axis (return two possible rotations)
        """
        # if euler[0] < 0:
        #     euler[0] = -np.pi/2
        # else:
        #     euler[0] = np.pi/2
        print("euler", euler)
        euler[2] = euler[0].copy()
        euler[0] = -np.pi/2

        pose_rotation_1 = Rotation.from_euler('xyz', euler).as_matrix()

        # pose_rotation_2 = Rotation.from_euler('xyz', euler).as_matrix()

        return pose_rotation_1#, pose_rotation_2

    def predict(self, img_path, pose_gt):

        img = self.image_reader(img_path)
        # scale_gt = 1e-3

        # theta = self.joint_pose_result[0].item() + np.pi/2
        # theta_z = np.pi
        # R_z = torch.tensor([
        #     [np.cos(theta_z), -np.sin(theta_z), 0],
        #     [np.sin(theta_z),  np.cos(theta_z), 0],
        #     [0,              0,             1]
        # ], device='cuda:0')

        # R_x = torch.tensor([
        #     [1, 0, 0],
        #     [0, np.cos(theta), -np.sin(theta)],
        #     [0, np.sin(theta), np.cos(theta)]
        # ], device='cuda:0')

        # ## ------ Configure Camera Parameters ------ ##
        # self.pose = np.eye(4)
        # self.pose[:3, :3] = pose_gt[:3, :3]
        # self.pose[:3, 3] = scale_gt * pose_gt[:3, 3]
        self.pose = self.get_posenet_prediction(img)

        ## decompose the pose to translation and rotation
        pose_rotation = self.pose[:3, :3]

        ## turn the rotation matrix to euler angles
        euler = Rotation.from_matrix(pose_rotation).as_euler('xyz')
        
        pose_rot_1, pose_rot_2 = None, None

        if np.abs(euler[0]) < np.pi/4 or np.abs(euler[0]) > 3 * np.pi/4:
            print("\ninhibiting x rotation")
            # pose_rot_1, pose_rot_2 = self.inhibit_x_rotation(euler)
            pose_rot_1 = self.inhibit_x_rotation(euler)

        if pose_rot_1 is not None:# and pose_rot_2 is not None:
            best_loss = 1000
            count = 0
            # for pose_rot in [pose_rot_1, pose_rot_2]:
            pose_rot = pose_rot_1
            self.pose[:3, :3] = pose_rot
            T_rotated_torch = torch.tensor(self.pose, dtype=self.camera.world_view_transform.dtype).transpose(0, 1).to(self.camera.data_device)
            self.camera.world_view_transform = T_rotated_torch

            ## ------ Differentiable Rendering ------ ##

            self.camera.joint_pose = torch.tensor(self.home_pose,).requires_grad_(True)
            self.camera.world_view_transform.requires_grad_(True) 

            self.joint_pose_result = torch.tensor(self.home_pose, requires_grad=True, device="cuda")
            self.world_view_transform_result = self.camera.world_view_transform.clone().detach().to("cuda").requires_grad_(True)
            # self.rotation_result = self.camera.world_view_transform[:3, :3].clone().detach().to("cuda").requires_grad_(True)
            # self.translation_result = self.camera.world_view_transform[3, :3].clone().detach().to("cuda").requires_grad_(True)
            # self.optimizer = torch.optim.Adam([
            #     {"params": self.joint_pose_result, "lr": self.joint_pose_lr},
            #     {"params": self.rotation_result, "lr": self.rotation_lr},
            #     {"params": self.translation_result, "lr": self.translation_lr}
            # ])

            self.optimizer = torch.optim.Adam([
                {"params": self.joint_pose_result, "lr": self.joint_pose_lr},
                {"params": self.world_view_transform_result, "lr": self.translation_lr},
            ])


            result = render(self.camera, self.gaussians, self.background_color)

            # Use the raw alpha channel as a continuous mask
            mask = result['render']  # Continuous alpha values, directly from render_alphas

            plt.figure()

            # Display the mask (optional, normalized for visualization)
            Utils.display_render(torch.clamp(mask, 0, 1))

            # self.video, self.joint_pose_result, self.rotation_result, self.translation_result, self.pc, losses= Utils.optimization_w2v_rot(self.camera, self.gaussians, self.background_color, self.reference_image, 
            #                                                                                self.optimizer, self.joint_pose_result, self.rotation_result, self.translation_result, self.num_iterations, plot=True)

            self.video, self.joint_pose_result, self.world_view_transform_result, self.pc, losses = Utils.optimization_w2v(
            self.camera, self.gaussians, self.background_color, self.reference_image, self.loss_fn, self.optimizer,
            self.joint_pose_result, self.world_view_transform_result, num_iterations=self.num_iterations, plot=False
            )


            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_rot = self.rotation_result.clone().detach()
                best_trans = self.translation_result.clone().detach()
                best_joint = self.joint_pose_result.clone().detach()
                best_pc = self.pc.clone().detach()
                best_video = self.video
                count += 1

            print("best estimation", count)
            

        else:
            best_rot, best_trans, best_joint, best_pc, best_video = None, None, None, None, None

            T_rotated_torch = torch.tensor(self.pose, dtype=self.camera.world_view_transform.dtype).transpose(0, 1).to(self.camera.data_device)
            self.camera.world_view_transform = T_rotated_torch


            ## ------ Differentiable Rendering ------ ##

            self.camera.joint_pose = torch.tensor(self.home_pose,).requires_grad_(True)
            self.camera.world_view_transform.requires_grad_(True) 

            self.joint_pose_result = torch.tensor(self.home_pose, requires_grad=True, device="cuda")  # Initial joint pose
            self.world_view_transform_result = self.camera.world_view_transform.clone().detach().to("cuda").requires_grad_(True) # Initial camera transform
            # self.rotation_result = self.camera.world_view_transform[:3, :3].clone().detach().to("cuda").requires_grad_(True)
            # self.translation_result = self.camera.world_view_transform[3, :3].clone().detach().to("cuda").requires_grad_(True)

            # self.optimizer = torch.optim.Adam([
            # {"params": self.joint_pose_result, "lr": self.joint_pose_lr},
            # {"params": self.rotation_result, "lr": self.rotation_lr},
            # {"params": self.translation_result, "lr": self.translation_lr}
            # ])
            self.optimizer = torch.optim.Adam([
                {"params": self.joint_pose_result, "lr": self.joint_pose_lr},
                {"params": self.world_view_transform_result, "lr": self.translation_lr},
            ])


            result = render(self.camera, self.gaussians, self.background_color)

            # Use the raw alpha channel as a continuous mask
            mask = result['render']  # Continuous alpha values, directly from render_alphas

            plt.figure()

            # Display the mask (optional, normalized for visualization)
            Utils.display_render(torch.clamp(mask, 0, 1))


            print("\n------------------- Optimizing -------------------\n")

            self.video, self.joint_pose_result, self.world_view_transform_result, self.pc, losses = Utils.optimization_w2v(
                self.camera, self.gaussians, self.background_color, self.reference_image, self.loss_fn, self.optimizer,
                self.joint_pose_result, self.world_view_transform_result, num_iterations=self.num_iterations, plot=False
            )


            
            # self.video, self.joint_pose_result, self.rotation_result, self.translation_result, self.pc, losses = Utils.optimization_w2v_rot(self.camera, self.gaussians, self.background_color, self.reference_image, 
            #                                                                                     self.optimizer, self.joint_pose_result, self.rotation_result, self.translation_result, self.num_iterations, plot=True)

        # Final results
        # self.world_view_transform_result = torch.eye(4, device="cuda")
        # self.world_view_transform_result[:3, :3] = self.rotation_result if best_rot is None else best_rot
        # self.world_view_transform_result[3, :3] = self.translation_result if best_trans is None else best_trans
        self.camera.world_view_transform = self.world_view_transform_result
        self.camera.joint_pose = best_joint if best_rot is None else best_joint
        self.pc = self.pc if best_pc is None else best_pc
        self.video = self.video if best_video is None else best_video
        # result = render(self.camera, self.gaussians, self.background_color)

        # # Use the raw alpha channel as a continuous mask
        # mask = result['render']  # Continuous alpha values, directly from render_alphas

        # plt.figure()

        # # Display the mask (optional, normalized for visualization)
        # Utils.display_render(torch.clamp(mask, 0, 1))


        tool_pose_2d, T_tc = self.convert_transformation()

        # Utils.make_video(self.video, filename="pose_est_init", fps=10)
        # IPImage(filename="pose_est_init.gif")
        original_image_width = 960
        original_image_height = 540
        tool_poses_init = tool_pose_2d
        scale = original_image_width / self.image_width
        offset = (original_image_width - original_image_height) /2
        starting_point = [tool_poses_init[0] * scale, tool_poses_init[1] * scale]
        projected_point = [starting_point[0], starting_point[1] - offset]


        return T_tc, projected_point



# def main(args):


    # for i in range(args.num_samples):




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='output/LND_short')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--model_num', type=int, default=20000)
    parser.add_argument('--model_name', type=str, default='LND_pose_train')

    args = parser.parse_args()

    # main(args)