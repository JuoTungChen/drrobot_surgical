import numpy as np
# from evaluate import Evaluator
import os
from natsort import natsorted
import argparse
import cv2
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from notebooks.utils import Utils
import torch
from moviepy import ImageSequenceClip
from skimage import measure
import tqdm

# from utils import quaternion_to_rotation_matrix
# path_to_dr_robot = "/home/iulian/chole_ws/src/drrobot"

# if path_to_dr_robot:
#     sys.path.append(os.path.join(path_to_dr_robot))

from estimator import PoseEstimator

# from visualization import show_blend_mask, show_axis, show_mask

TASK_CHOICES = {'l': 'LND', 'm': 'MBF'}


def get_data_samples(root_path):
    image_paths = natsorted(os.listdir(os.path.join(root_path,'seg_masks')))
    # image_paths = natsorted(os.listdir(os.path.join(root_path,'image')))
    # pose_paths = natsorted(os.listdir(os.path.join(root_path,'pose')))
    image_paths = [os.path.join(root_path,'seg_masks',img_path) for img_path in image_paths]
    # image_paths = [os.path.join(root_path,'image',img_path) for img_path in image_paths]
    # pose_paths = [os.path.join(root_path,'pose',pose_path) for pose_path in pose_paths]
    # mask_path = os.path.join(root_path,'mask')    #in case of someone use the masks

    return image_paths

def invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = - R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, [-1]] = t_inv
    return T_inv

def make_video_overlay(video1, video2, filename, fps=30):
    video1 = Utils.process_video(video1)
    video2 = Utils.process_video(video2)
    video = []
    for i in range(len(video1)):
        overlay = cv2.addWeighted(video1[i], 0.3, video2[i], 0.8, 0)
        rendered_mask = np.any(video1[i] > 0.3, axis=-1)  # Threshold for silhouette detection
        contours = measure.find_contours(rendered_mask.astype(float), 0.8)
        # Draw the contours on the overlay
        for contour in contours:
            overlay[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)] = [255,0,0]

        video.append(overlay)

    clip = ImageSequenceClip(list(video), fps=fps)
    clip.write_gif(f"{filename}.gif")


def get_args():
    parser = argparse.ArgumentParser(description='Evaluateion script for SurgRIPE.')
    parser.add_argument('--path', help= 'Get path to data root path.')
    parser.add_argument('--type', choices=TASK_CHOICES.keys(), default='l', help= 'Instrument Type for test.')
    parser.add_argument('--model_num', type=int, default=105000)
    parser.add_argument('--model_name', type=str, default='LND_pose_train_1')
    return parser.parse_args()



def compare_traj(img_path, T_tc_homogeneous_list, T_tc_list, img_height=256,
                 ori_img_height=540):

    ## read the csv file
    ee_csv_path = img_path + "/ee_csv.csv"

    ee_csv = pd.read_csv(ee_csv_path)
    header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                            "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                            "psm1_jaw"]

    header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                            "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                            "psm2_jaw"]

    ee_l_qpos = ee_csv[header_name_qpos_psm1].to_numpy()

    # ee_l_qpos = ee_csv.to_numpy()

    ee_l_qpos_init = ee_l_qpos[0]
    R_ee_init = Rotation.from_quat(ee_l_qpos[0][3:7]).as_matrix()
    T_ee_init = ee_l_qpos[0][:3]
    H_ee_init = np.eye(4)
    H_ee_init[:3, :3] = R_ee_init
    H_ee_init[:3, 3] = T_ee_init
    print(H_ee_init)
    # ## calculate the relative action trajectory against the first frame of the ground truth trajectory
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # theta = np.pi  
    # R_z = np.array([
    #     [np.cos(theta), -np.sin(theta), 0],
    #     [np.sin(theta),  np.cos(theta), 0],
    #     [0,              0,             1]
    # ])

    # results = []

    # H_tc_init = T_tc_homogeneous_list[0].T
    # print(len(T_tc_homogeneous_list))
    # euler_tc = Rotation.from_matrix(H_tc_init[:3, :3]).as_euler('zyx', degrees=True)
    # # print(euler_tc)
    # count = 0
    # scale = img_height/ori_img_height
    # H_result = []
    # count = 0
    # for H_est in T_tc_homogeneous_list:
    #     H_est_relative = np.linalg.inv(H_tc_init) @ H_est.T
    #     res = H_est_relative
    #     T_relative = scale * (T_tc_list[count] - T_tc_list[0])
    #     res[:3, 3] = R_z @ T_relative
    #     results.append([res[0, 3], res[1, 3], res[2, 3]])
    #     H_result.append(res)
    #     count += 1

    # results = np.array(results)  # Convert list to NumPy array

    # ax.scatter(results[:, 0], results[:, 1], results[:, 2], c="b", label="estimated")

    # ee_l_qpos_relative = ee_l_qpos - ee_l_qpos_init

    # ax.scatter(ee_l_qpos_relative[:, 0], ee_l_qpos_relative[:, 1], ee_l_qpos_relative[:, 2], c="r", label="forward kinematics")

    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    # ax.set_xlim(-0.005, 0.015)
    # ax.set_ylim(-0.01, 0.01)
    # ax.set_zlim(-0.01, 0.01)
    # ## look from the top
    # ax.view_init(-90, 60)
    # ax.legend(fontsize=12)
    # # ax.view_init(0, 0)
    # ## set title
    # # plt.title("relative action trajectory", fontsize=16)
    # # ax.legend(["robot", "human"])
    # plt.show()

def save_to_csv(dir, joint_pose_results, T_tc_homogeneous_list, T_tc_list, img_height=256,
                 ori_img_height=540):
    theta = np.pi  
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    results = []

    init = np.array(
        [[ 0.51198019,  0.43195199,  0.74249159,  0.00782541],                                                                                               
        [ 0.03269589 , 0.85394442, -0.51933603, -0.1376347 ],
        [-0.85837478 , 0.29016618,  0.42307958,  0.19184512],
        [ 0.         , 0.,          0.,          1.        ]]
        )

    H_tc_init = T_tc_homogeneous_list[0].T
    # print(len(T_tc_homogeneous_list))
    euler_tc = Rotation.from_matrix(H_tc_init[:3, :3]).as_euler('zyx', degrees=True)
    # print(euler_tc)
    count = 0
    scale = img_height/ori_img_height
    H_result = []
    count = 0
    jaw = [0.0] * (len(joint_pose_results) + 1)

    for H_est in T_tc_homogeneous_list:
        H_est_relative = np.linalg.inv(H_tc_init) @ H_est.T
        res = H_est_relative
        # res = H_est
        # res[:3, :3] = R_z @ H_est[:3, :3]
        T_relative = scale * (T_tc_list[count] - T_tc_list[0])
        res[:3, 3] = R_z @ T_relative
        results.append([res[0, 3], res[1, 3], res[2, 3]])
        H_result.append(res)
        jaw[count + 1] = -joint_pose_results[count][1].item() - joint_pose_results[count][2].item()
        count += 1

    results = np.array(results)  # Convert list to NumPy array

    csv_result = []
    count = 0
    # print(H_result[0])
    init_quat = Rotation.from_matrix(init[:3, :3]).as_quat()
    prev_translation = init[:3, 3]
    prev_quat = init_quat
    for n, r in enumerate(H_result):
        translation_result = r[:3, 3] + init[:3, 3]
        rotation_result = init[:3, :3] @ r[:3, :3]

        quat_results = Rotation.from_matrix(rotation_result).as_quat()


        csv_result.append(np.concatenate([prev_translation, prev_quat, [jaw[n]], translation_result, quat_results, [jaw[n+1]],
                                          [0,0,0,1,0,0,0,0],
                                          [0,0,0,1,0,0,0,0]]))            
        prev_translation = translation_result
        prev_quat = quat_results

    ## write the results to csv file
    results = pd.DataFrame(csv_result, columns=["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                            "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                            "psm1_jaw", 
                            "psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                            "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                            "psm1_jaw_sp",
                            "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                            "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                            "psm2_jaw",
                            "psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                            "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                            "psm2_jaw_sp",
                            ])



    results.to_csv(f"{dir}/ee_estimate.csv", index=False)


def main():
    args = get_args()
    instrument_type = TASK_CHOICES.get(args.type)
    pose_estimator = PoseEstimator(args.model_name, args.model_num)
    root_path = args.path    #the root path for dataset e.g. .../LND/TRAIN
    first_run=True
    root_path = natsorted(os.listdir(root_path))
    root_path = [os.path.join(args.path, root) for root in root_path]
    
    for dir in tqdm.tqdm(root_path):

        T_tc_homogeneous_list = []
        T_tc_list = []
        video = []
        joint_pose_results = []
        world_view_transform_results = []
        pc_results = {}
        n = 0
        test_samples = get_data_samples(dir)
        imgs = []
        # evaluator = Evaluator(root_path, instrument_type)
        # print(pose_estimator.P.T.detach().cpu().numpy().astype(np.float64))
        first_img = True
        # image_path = "/home/iulian/chole_ws/data/tissue_lift/tissue_1/tissue_lift/20250226-122933-075955/seg_masks/"
        # test_samples = natsorted(os.listdir(image_path))
        # test_samples = [os.path.join(image_path, img) for img in test_samples]
        for n, test_sample in enumerate(test_samples):

            image_path = test_sample

            if first_img and first_run:
                # print("\nfirst run\n")
                print(image_path)

                pose_pred, projected_point, last_pose, joint_p = pose_estimator.predict_rand_init(image_path, first_run, first_img)
                pose_estimator.display_overlay()

                first_run=False
                first_img = False

            elif first_img:
                # print("\nfirst frame\n")
                pose_pred, projected_point, last_pose, joint_p = pose_estimator.predict_rand_init(image_path, first_run, first_img)
                first_img = False

            else:
                _, img = pose_estimator.image_reader(image_path)
                imgs.append(img)

            # pose_estimator.display_overlay()
        joint_pose_results, world_view_transform_results, pc_results, videos = pose_estimator.loop_predict(imgs)
        # print("joint_pose_results", joint_pose_results)
        # print("world_view_transform_results", world_view_transform_results)
        ## make the video into gif
        # Utils.make_video(videos, filename="pose_video", fps=10)
        # Utils.make_video(imgs, filename="imgs", fps=10)
        # make_video_overlay(videos, imgs, filename="epo_overlay")

        image_width = 256
        image_height = 256


        for n, w in enumerate(world_view_transform_results):

            ## calculate middle point
            # middle_point = (pc_results[n][min_z_idx] + pc_results[n][original_min_y_idx]) / 2
            screw_point = pc_results[n][pose_estimator.ee_point_idx]
            # print(screw_point)
            # print("screw_point", screw_point)
            theta_z = - np.pi /2 
            R_z = torch.tensor([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z),  np.cos(theta_z), 0],
                [0,              0,             1]
            ], device='cuda:0')

            theta = joint_pose_results[n][0].item() - np.pi/2
            R_x = torch.tensor([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ], device='cuda:0')
            
            ori_rotation = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device='cuda:0')

            middle_point_homogeneous = torch.cat([torch.tensor(screw_point, device='cuda:0'), torch.tensor([1.0], device='cuda:0')])
            T_wt_homogeneous = torch.eye(4, device='cuda:0')
            T_wt_homogeneous[:3, :3] = ori_rotation
            # T_wt_homogeneous[:3, :3] = R_z @ R_x

            T_wt_homogeneous[:3, 3] = middle_point_homogeneous[:3]

            T_wc = w  #.transpose(1, 0)

            # Calculate the position of the surgical tool in the camera frame
            T_tc_homogeneous = T_wt_homogeneous.T @ T_wc


            projected_pose = pose_estimator.P.T.detach().cpu() @ T_tc_homogeneous[3, :].detach().cpu()

            # Normalize by the homogeneous coordinate (w) to get 2D image coordinates

            x_2d = ((projected_pose[0] / projected_pose[2] + 1) * 0.5) * image_width
            y_2d = ((projected_pose[1] / projected_pose[2] + 1) * 0.5) * image_height

            tool_pose = (x_2d, y_2d)

            T_tc_list.append(T_tc_homogeneous[3, :3].detach().cpu().numpy())
            T_tc_homogeneous_list.append(T_tc_homogeneous.clone().detach().cpu().numpy())
            # tool_poses.append(tool_pose)

        
        # compare_traj(dir, T_tc_homogeneous_list, T_tc_list)

        save_to_csv(dir, joint_pose_results, T_tc_homogeneous_list, T_tc_list)
        # input("Press Enter to continue...")

        ### mm to m
        # # im = show_axis(image, pose_pred[:3,:3].astype(np.float64), pose_pred[:3,-1].astype(np.float64), evaluator.K.astype(np.float64), None, 6, point=projected_point, is_show=True)
        # # im = show_axis(image, pose_pred[:3,:3].astype(np.float64), pose_pred[:3,-1].astype(np.float64), pose_estimator.P.T.detach().cpu().numpy().astype(np.float64), None, 1, True)
        # im = im.astype(np.uint8)  # Ensure the image is in the correct format for OpenCV
        # # print("Returned image dimensions:", im.shape)
        # print("\nground truth\n", pose_gt)
        # print("\npredicted:\n",pose_pred[:3])
        # # evaluator.projection_2d(pose_pred[:3], pose_gt, evaluator.K)
        # evaluator.trans_rot_error(pose_pred[:3], pose_gt)
        # evaluator.evaluate(pose_gt, pose_pred[:3])

    # evaluator.summarize()


if __name__ == "__main__":
    main()



