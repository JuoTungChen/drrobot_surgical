import os
from contextlib import redirect_stdout
from video_api import initialize_gaussians
from gaussian_renderer import render
import sys 
import torch 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
from skimage import measure
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from scipy.linalg import expm  # Matrix exponential for SO(3) updates
from pytorch_msssim import ssim
from torch.linalg import matrix_exp
import tqdm


class Utils:
    def __init__(self):
        if 'notebooks' not in os.listdir(os.getcwd()):
            os.chdir('../') #changing directories so that output/gsplat_full etc. exists

    @staticmethod
    def display_render(image_tensor: torch.Tensor):
        im = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
        im = (im * 255).astype(np.uint8)
        plt.imshow(im)
        plt.axis('off')
        plt.show()

    @staticmethod
    def pad_to_match_aspect(image, target_aspect_ratio):
        _, h, w = image.shape
        current_aspect_ratio = w / h

        if current_aspect_ratio < target_aspect_ratio:
            new_width = int(h * target_aspect_ratio)
            pad = (new_width - w) // 2
            padding = (pad, pad, 0, 0)
        else:
            new_height = int(w / target_aspect_ratio)
            pad = (new_height - h) // 2
            padding = (0, 0, pad, pad)
        
        return F.pad(image.unsqueeze(0), padding, mode='constant', value=0).squeeze(0)

    @staticmethod
    def enforce_homogeneous_transform(matrix):
        with torch.no_grad():
            rotation = matrix[:3, :3]
            u, _, v = torch.svd(rotation)
            rotation_orthogonal = torch.mm(u, v.t())
            matrix[:3, :3] = rotation_orthogonal
            matrix[:, 3] = torch.tensor([0, 0, 0, 1], device=matrix.device)
        return matrix

    # @staticmethod
    # def enforce_homogeneous_transform(matrix):
    #     rotation = matrix[:3, :3]
    #     u, _, v = torch.svd(rotation)  # Ensure rotation matrix is in SO(3)
    #     rotation_orthogonal = torch.mm(u, v.t())  # Differentiable projection onto SO(3)
        
    #     # Ensure homogeneous transformation without detaching gradients
    #     matrix_homo = matrix.clone()
    #     matrix_homo[:3, :3] = rotation_orthogonal
    #     matrix_homo[3, :3] = matrix[3, :3]  # Keep translation unchanged
    #     matrix_homo[:, 3] = torch.tensor([0, 0, 0, 1], dtype=matrix.dtype, device=matrix.device)
        
    #     return matrix_homo


    @staticmethod
    def enforce_homogeneous_transform_1(matrix):
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        U, _, Vt = torch.linalg.svd(R, full_matrices=False)
        R_orthonormal = U @ Vt
        if torch.det(R_orthonormal) < 0:
            R_orthonormal[:, -1] *= -1
        T_homogeneous = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
        T_homogeneous[:3, :3] = R_orthonormal.T
        T_homogeneous[3, :3] = t
        return T_homogeneous

    @staticmethod
    def decompose_homogeneous_matrix(matrix):
        rotation_matrix = matrix[:3, :3].T.cpu().numpy()
        translation = matrix[3, :3]
        euler_angles = torch.tensor(Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False), device=matrix.device)
        return euler_angles, translation

    @staticmethod
    def assemble_homogeneous_matrix(euler_angles, translation):
        rotation_matrix = torch.tensor(Rotation.from_euler('xyz', euler_angles.detach().cpu().numpy(), degrees=False).as_matrix(), device=euler_angles.device)
        matrix = torch.eye(4, device=euler_angles.device)
        matrix[:3, :3] = rotation_matrix.T
        matrix[3, :3] = translation
        return matrix

    @staticmethod
    def decompose_homogeneous_matrix_quaternion(matrix):
        rotation_matrix = matrix[:3, :3].T
        translation = matrix[3, :3]
        quaternion = torch.tensor(Rotation.from_matrix(rotation_matrix.cpu().numpy()).as_quat(), device=matrix.device, dtype=torch.float32)
        return quaternion, translation

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        q0, q1, q2, q3 = quaternion
        R = torch.stack([
            torch.tensor([1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)], device=quaternion.device),
            torch.tensor([2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)], device=quaternion.device),
            torch.tensor([2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)], device=quaternion.device)
        ])
        return R

    @staticmethod
    def assemble_homogeneous_matrix_quaternion(quaternion, translation):
        rotation_matrix = Utils.quaternion_to_rotation_matrix(quaternion)
        matrix = torch.eye(4, device=quaternion.device)
        matrix[:3, :3] = rotation_matrix.T
        matrix[3, :3] = translation
        return matrix

    @staticmethod
    def enforce_joint_limit(joint_pose):
        joint_pose = joint_pose.clone()  # Create a copy to avoid modifying the original tensor
        joint_pose[2] = torch.minimum(joint_pose[2], torch.tensor(0.47, device=joint_pose.device))
        joint_pose[4] = torch.minimum(joint_pose[4], torch.tensor(0.42, device=joint_pose.device))
        return joint_pose


    @staticmethod
    def optimization(camera, gaussians, background_color, reference_image, loss_fn, optimizer, joint_pose_result, world_view_transform_result, euler_angles, translation, num_iterations=200):
        video = []
        losses = []
        for i in range(num_iterations):
            world_view_transform_result = Utils.assemble_homogeneous_matrix(euler_angles, translation)
            # joint_pose_result = Utils.enforce_joint_limit(joint_pose_result)
            camera.joint_pose = joint_pose_result
            camera.world_view_transform = world_view_transform_result
            rendered_image = torch.clamp(render(camera, gaussians, background_color)['render'], 0, 1)
            loss = loss_fn(rendered_image, reference_image)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")
                frame = rendered_image.detach().cpu()
                video.append(frame)
        return video, joint_pose_result, Utils.assemble_homogeneous_matrix(euler_angles.detach(), translation.detach()), euler_angles, translation

    @staticmethod
    def optimization_w2v(camera, gaussians, background_color, reference_image, loss_fn, optimizer, joint_pose_result, world_view_transform_result, num_iterations=200, plot=False):
        video = []
        losses = []
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

        stopping_count = 0
        normalization_factor = torch.sum(reference_image ** 2).item()
        normalized_threshold = 1e-6 * normalization_factor
        previous_loss = None
        loss_change_threshold = 1e-7
        init_world_view_transform = world_view_transform_result.clone()
        for i in tqdm.tqdm(range(num_iterations)):
            # joint_pose_result = Utils.enforce_joint_limit(joint_pose_result)
            camera.joint_pose = joint_pose_result
            camera.world_view_transform = world_view_transform_result
            render_result = render(camera, gaussians, background_color)
            rendered_image = torch.clamp(render_result['render'], 0, 1)
            pc = render_result['pc']
            loss = loss_fn(rendered_image, reference_image)
            losses.append(loss.item())
            if previous_loss is not None:
                loss_change = abs(previous_loss - loss.item())
                if loss_change < loss_change_threshold:
                    stopping_count += 1
                    # print(f"Loss change is less than threshold. Count = {stopping_count}")
            if stopping_count == 10:
                print("Early stopping at", i)
                break
            previous_loss = loss.item()

            ## check gradient
            # print("Gradients for joint", joint_pose_result.grad)
            # print("Gradients for world_view_transform", world_view_transform_result.grad)


            optimizer.zero_grad()
            loss.backward()
            scheduler.step(loss)

            with torch.no_grad():
                rotation = world_view_transform_result[:3, :3]
                translation = world_view_transform_result[3, :3]
                # Apply gradients
                rotation_grad = world_view_transform_result.grad[:3, :3]
                translation_grad = world_view_transform_result.grad[:3, 3]

                rot_learning_rate = 0.3  # Example learning rate
                trans_learning_rate = 3e-4  # Example learning rate
                rotation_update = torch.eye(3, device=rotation.device) + rot_learning_rate * rotation_grad
                rotation = torch.mm(rotation.T, rotation_update)
                translation_grad = torch.clamp(translation_grad, -0.02, 0.02)  # Prevent extreme shifts

                translation = translation - trans_learning_rate * translation_grad

                # Recompose the matrix
                world_view_transform_result[:3, :3] = rotation.T
                world_view_transform_result[3, :3] = translation

            optimizer.step()
            with torch.no_grad():
                world_view_transform_result.copy_(Utils.enforce_homogeneous_transform(world_view_transform_result))

            if i % 10 == 0:
                video.append(rendered_image.detach().cpu())
        if plot:
            Utils.plot_loss(losses)

        ## calculate difference between initial and final world_view_transform (element wise)
        diff = torch.abs(init_world_view_transform - world_view_transform_result)
        print("Difference between initial and final world_view_transform:\n", diff)


        return video, joint_pose_result, world_view_transform_result, pc, losses




    @staticmethod
    def optimization_w2v_rot(camera, gaussians, background_color, reference_image, optimizer, 
                        joint_pose_result, rotation_result, translation_result, num_iterations=200, plot=False):

        video = []
        losses = []
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
        stopping_count = 0
        normalization_factor = torch.sum(reference_image ** 2).item()
        normalized_threshold = 1e-6 * normalization_factor
        previous_loss = None
        loss_change_threshold = 1e-6
        init_rotation = rotation_result.clone()
        init_translation = translation_result.clone()
        world_view_transform_result = torch.eye(4, device="cuda")
        # loss_fn = torch.nn.MSELoss()
        # Loss function combining SSIM and MSE
        def combined_loss(rendered, reference, alpha=0.8):
            mse = F.mse_loss(rendered, reference)
            
            # Ensure inputs are 4D: (batch, channels, height, width)
            rendered = rendered.unsqueeze(0) if rendered.dim() == 3 else rendered
            reference = reference.unsqueeze(0) if reference.dim() == 3 else reference
            
            ssim_l = 1 - ssim(rendered, reference, data_range=1.0)
            return alpha * ssim_l + (1 - alpha) * mse

        # progree bar
        for i in tqdm.tqdm(range(num_iterations)):
            camera.joint_pose = joint_pose_result
            camera.world_view_transform = torch.eye(4, device="cuda")
            camera.world_view_transform[:3, :3] = rotation_result
            camera.world_view_transform[3, :3] = translation_result

            render_result = render(camera, gaussians, background_color)
            rendered_image = torch.clamp(render_result['render'], 0, 1)
            pc = render_result['pc']

            loss = combined_loss(rendered_image, reference_image)
            losses.append(loss.item())

            ## early stopping
            if previous_loss is not None:
                loss_change = abs(previous_loss - loss.item())
                if loss_change < loss_change_threshold:
                    stopping_count += 1

            if stopping_count == 50:
                print("Early stopping at", i)
                break
            previous_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            scheduler.step(loss)


            with torch.no_grad():
                # Rotation update using skew-symmetric matrix
                omega = rotation_result.grad.reshape(-1)[:3]  # Ensure it's a (3,) vector
                skew_sym = torch.tensor([[0, -omega[2], omega[1]],
                                        [omega[2], 0, -omega[0]],
                                        [-omega[1], omega[0], 0]], dtype=torch.float32, device=rotation_result.device)

                # Apply matrix exponential map (torch version, avoids SciPy)
                rotation_update = matrix_exp(skew_sym)

                # Update rotation matrix
                rotation_result.copy_(rotation_update @ rotation_result)

                # Apply translation update with gradient clamping
                translation_grad = translation_result.grad
                translation_grad = torch.clamp(translation_grad, -0.01, 0.01)  # Prevent extreme shifts
                translation_result -= optimizer.param_groups[2]['lr'] * translation_grad

            optimizer.step()

            if i % 10 == 0:
                video.append(rendered_image.detach().cpu())

        if plot:
            Utils.plot_loss(losses)

        # Compute the difference between initial and final transformation
        rotation_diff = torch.abs(init_rotation - rotation_result)
        translation_diff = torch.abs(init_translation - translation_result)
        print("Difference in rotation:\n", rotation_diff)
        print("Difference in translation:\n", translation_diff)

        return video, joint_pose_result, rotation_result, translation_result, pc, losses



    @staticmethod
    def optimization_w2v_test(camera, gaussians, background_color, reference_image, loss_fn, optimizer, joint_pose_result, world_view_transform_result, num_iterations=200, plot=False):
        video = []
        losses = []
        
        stopping_count = 0
        normalization_factor = torch.sum(reference_image ** 2).item()
        normalized_threshold = 1e-6 * normalization_factor
        previous_loss = None
        loss_change_threshold = 1e-7
        for i in range(num_iterations):
            # joint_pose_result = Utils.enforce_joint_limit(joint_pose_result)
            camera.joint_pose = joint_pose_result
            camera.world_view_transform = world_view_transform_result
            render_result = render(camera, gaussians, background_color)
            rendered_image = torch.clamp(render_result['render'], 0, 1)
            pc = render_result['pc']
            loss = loss_fn(rendered_image, reference_image)
            losses.append(loss.item())
            if previous_loss is not None:
                loss_change = abs(previous_loss - loss.item())
                if loss_change < loss_change_threshold:
                    stopping_count += 1
                    # print(f"Loss change is less than threshold. Count = {stopping_count}")
            if stopping_count == 50:
                print("Early stopping at", i)
                break
            previous_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                world_view_transform_result.copy_(Utils.enforce_homogeneous_transform(world_view_transform_result))
            if i % 10 == 0:
                video.append(rendered_image.detach().cpu())
        if plot:
            Utils.plot_loss(losses)
        return video, joint_pose_result, world_view_transform_result, pc, losses[-1], losses[0]


    @staticmethod
    def matrix_to_state(T):
        """Converts a 4x4 transformation matrix into a 13D state (xyz, quaternion, linear & angular velocity)."""
        translation = T[:3, 3]
        rotation = R.from_matrix(T[:3, :3]).as_quat()  # Quaternion (x, y, z, w)
        velocity = np.zeros(3)  # Initialize velocity (vx, vy, vz)
        angular_velocity = np.zeros(3)  # Initialize angular velocity (wx, wy, wz)
        return np.hstack((translation, rotation, velocity, angular_velocity))

    @staticmethod
    def state_to_matrix(state):
        """Converts a 13D state (xyz, quaternion) back to a 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(state[3:7]).as_matrix()  # Convert quaternion to rotation matrix
        T[:3, 3] = state[:3]  # Assign translation
        return T

    @staticmethod
    def quaternion_exp(omega, dt):
        """Computes quaternion update using angular velocity and small-angle approximation."""
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-6:
            return np.array([0, 0, 0, 1])  # No rotation if angular velocity is too small
        axis = omega / np.linalg.norm(omega)
        delta_q = R.from_rotvec(axis * theta).as_quat()
        return delta_q

    @staticmethod
    def integrate_rotation(q, omega, dt):
        """Applies angular velocity update to quaternion."""
        delta_q = Utils.quaternion_exp(omega, dt)
        q_new = R.from_quat(q) * R.from_quat(delta_q)
        return q_new.as_quat()

    @staticmethod
    def init_kalman_filter(initial_T, dt=0.01):
        """Initializes a Kalman filter for 6D pose estimation with translation & rotation velocity."""
        state_dim = 13  # (x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)
        measurement_dim = 7  # (x, y, z, qx, qy, qz, qw)

        kf = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)

        initial_state = Utils.matrix_to_state(initial_T)
        kf.x = initial_state  # Initial state (position, quaternion, zero velocities)

        # State transition matrix (for constant velocity model)
        kf.F = np.eye(state_dim)
        for i in range(3):  # Position update: x += v * dt
            kf.F[i, i + 7] = dt
        for i in range(3, 7):  # Quaternion update (handled separately)
            kf.F[i, i] = 1

        kf.H = np.eye(measurement_dim, state_dim)  # Measurement function: observe (x, y, z, q)

        kf.P *= 1e-2  # Initial covariance
        kf.R *= 1e-3  # Measurement noise covariance
        kf.Q *= 1e-3  # Process noise covariance

        return kf

    @staticmethod
    def optimization_kalman(camera, gaussians, background_color, reference_image, loss_fn, optimizer, joint_pose_result, world_view_transform_result, num_iterations=200, plot=False, dt=0.01):
        video = []
        losses = []
        
        stopping_count = 0
        normalization_factor = torch.sum(reference_image ** 2).item()
        previous_loss = None
        loss_change_threshold = 1e-6

        # Convert initial world-view transform (4x4 matrix) to Kalman filter state
        kf = Utils.init_kalman_filter(world_view_transform_result.clone().detach().cpu().numpy().transpose(1, 0), dt)


        for i in range(num_iterations):
            # Predict next state
            kf.predict()

            # Get predicted transformation matrix from Kalman filter
            predicted_T = Utils.state_to_matrix(kf.x)

            # Update world-view transform result with the Kalman prediction
            with torch.no_grad():
                world_view_transform_result.copy_(torch.tensor(predicted_T.transpose(1, 0), dtype=torch.float32))
            # Render with predicted transform
            camera.world_view_transform = world_view_transform_result
            render_result = render(camera, gaussians, background_color)
            rendered_image = torch.clamp(render_result['render'], 0, 1)
            pc = render_result['pc']

            # Compute loss
            loss = loss_fn(rendered_image, reference_image)
            losses.append(loss.item())

            # Check for early stopping
            if previous_loss is not None and abs(previous_loss - loss.item()) < loss_change_threshold:
                stopping_count += 1
                # print(f"Loss change is below threshold. Count = {stopping_count}")
            if stopping_count == 10:
                print("Early stopping at", i)
                break
            previous_loss = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                world_view_transform_result.copy_(Utils.enforce_homogeneous_transform(world_view_transform_result))

            # Measurement update (get new estimate of pose from rendering)
            measured_T = world_view_transform_result.clone().detach().cpu().numpy().transpose(1, 0)
            measured_state = Utils.matrix_to_state(measured_T)
            kf.update(measured_state[:7])  # Update only position and quaternion

            # Update Kalman state (handle quaternion separately)
            kf.x[3:7] = Utils.integrate_rotation(kf.x[3:7], kf.x[10:13], dt)

        # if i % num_iterations == 0:
        video.append(rendered_image.detach().cpu())

        if plot:
            Utils.plot_loss(losses)
        return video, joint_pose_result, world_view_transform_result, pc


    @staticmethod
    def optimization_quat(camera, gaussians, background_color, reference_image, loss_fn, optimizer, joint_pose_result, world_view_transform_result, quaternion, translation, num_iterations=200, plot=False):
        video = []
        losses = []
        for i in range(num_iterations):
            camera.joint_pose = joint_pose_result
            camera.world_view_transform = world_view_transform_result
            rendered_image = torch.clamp(render(camera, gaussians, background_color)['render'], 0, 1)
            loss = loss_fn(rendered_image, reference_image)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            print("Gradients for joint", joint_pose_result.grad)
            print("Gradients for world_view_transform", world_view_transform_result.grad)
            print("Gradients for quaternion:", quaternion.grad)
            print("Gradients for translation:", translation.grad)
            optimizer.step()
            with torch.no_grad():
                quaternion /= torch.norm(quaternion)
            if i % 10 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")
                video.append(rendered_image.detach().cpu())
        if plot:
            Utils.plot_loss(losses)
        return video, joint_pose_result, Utils.assemble_homogeneous_matrix_quaternion(quaternion.detach(), translation.detach()), quaternion, translation

    @staticmethod
    def plot_loss(losses):
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs. Iteration')
        plt.show()

    @staticmethod
    def count_files(path):
        return len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])

    @staticmethod
    def make_video(video, filename, fps=30):
        video = torch.stack(video)
        video = torch.clamp(video, 0, 1)
        video = video.cpu().detach().numpy()
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video * 255).astype(np.uint8)
        clip = ImageSequenceClip(list(video), fps=fps)
        clip.write_gif(f"{filename}.gif")

    @staticmethod
    def process_video(video):
        video = torch.stack(video)
        video = torch.clamp(video, 0, 1)
        video = video.cpu().detach().numpy()
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video * 255).astype(np.uint8)
        return video

    ## make video that overlays the rendered image on the reference image
    @staticmethod
    def make_video_overlay(video1, video2, filename, fps=30):
        video1 = Utils.process_video(video1)
        video2 = Utils.process_video(video2)
        video = []
        for i in range(len(video1)):
            overlay = cv2.addWeighted(video1[i], 0.3, video2[i], 0.8, 0)
            rendered_mask = np.any(video1[i] > 0.08, axis=-1)  # Threshold for silhouette detection
            contours = measure.find_contours(rendered_mask.astype(float), 0.5)
            # Draw the contours on the overlay
            for contour in contours:
                overlay[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = [255, 0, 0]

            video.append(overlay)

        clip = ImageSequenceClip(list(video), fps=fps)
        clip.write_gif(f"{filename}.gif")

class KalmanFilter6D:
    def __init__(self, dt=0.01, process_noise=1e-4, measurement_noise=1e-2):
        self.dt = dt  # Time step

        # State vector [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        self.x = np.zeros((13, 1))

        # State transition matrix (F)
        self.F = np.eye(13)
        self.F[:3, 7:10] = np.eye(3) * dt  # Position update
        self.F[3:7, 10:13] = np.eye(4) * (0.5 * dt)  # Quaternion update approximation

        # Process noise covariance (Q)
        self.Q = np.eye(13) * process_noise

        # Measurement matrix (H)
        self.H = np.zeros((7, 13))
        self.H[:3, :3] = np.eye(3)  # Position measurement
        self.H[3:7, 3:7] = np.eye(4)  # Quaternion measurement

        # Measurement noise covariance (R)
        self.R = np.eye(7) * measurement_noise

        # Covariance matrix (P)
        self.P = np.eye(13)

    def predict(self):
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update step with measurement z (position + quaternion)."""
        # Measurement residual
        y = z - (self.H @ self.x)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (np.eye(13) - K @ self.H) @ self.P

        # Normalize quaternion
        q = self.x[3:7].flatten()
        self.x[3:7] = q / np.linalg.norm(q)

    def get_state(self):
        """Returns estimated position and quaternion."""
        pos = self.x[:3].flatten()
        quat = self.x[3:7].flatten()
        return pos, quat
    

"""
# Example usage:
kf = KalmanFilter6D()

# Simulated noisy pose measurement
true_pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])  # Position + Quaternion
noisy_measurement = true_pose + np.random.normal(0, 0.01, size=7)

kf.predict()
kf.update(noisy_measurement)

position, quaternion = kf.get_state()
print("Estimated Position:", position)
print("Estimated Quaternion:", quaternion)

"""