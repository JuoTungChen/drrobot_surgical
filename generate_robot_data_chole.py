import numpy as np
import os
import itertools
import ray
import time
from tqdm import tqdm
import glob
import mujoco
import shutil
from PIL import Image
from utils.mujoco_utils import compute_camera_extrinsic_matrix, compute_camera_intrinsic_matrix
from filelock import FileLock
from utils.mujoco_utils import get_canonical_pose, set_xml_light_params, find_non_collision_pose, save_robot_metadata, ControlRobot
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

"""
This file creates many Mujoco environments simultaneously and renders images from them. 
It is used to generate the point cloud dataset for the robot dataset.

Instructions for preparing a XML file for processing:
- Remove fixed joints from the model
"""


@ray.remote
class MujocoActor:
    """
    Converts a XML file into an image/depth/point cloud dataset for kinematically-aware gaussian training.
    """
    def __init__(self, 
                 actor_id,
                 model_xml_dir, 
                 save_dir,
                 diffuse_light_params=(0.5, 0.5, 0.5),
                 ambient_light_params=(0.5, 0.5, 0.5),
                 resolution=(256, 256),
                 pcd_max_points=20_000,
                 ):
        self.model_xml_dir = model_xml_dir
        self.model_xml_path = os.path.join(model_xml_dir, "scene.xml")
        self.robot_name = model_xml_dir.split('/')[-1]

        #modify xml file to have reasonable lighting params
        set_xml_light_params(self.model_xml_path, diffuse_light_params, ambient_light_params)

        #initiilize the mujoco environment in a fault-tolerant way
        attempt, MAX_ATTEMPTS = 0, 10
        success = False
        while not success and attempt < MAX_ATTEMPTS:
            try:
                self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)
                self.data = mujoco.MjData(self.model)
                self.save_dir = save_dir
                # self.random_base_placement()
                # self.look_at_end_effector()
                self.renderer = mujoco.Renderer(self.model, resolution[0], resolution[1])
                success = True
            except Exception as e:
                attempt += 1
                sleep_time = 1.5 ** attempt
                print(f"Attempt {attempt} failed with error: {e}. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
        if not success:
            raise Exception("Failed to initialize Mujoco components after multiple attempts.")

        self.pcd_max_points = pcd_max_points
        save_robot_metadata(self.model, self.model_xml_dir, self.save_dir)
        self.control_robot = ControlRobot(self.model, self.data, self.robot_name)


    def generate_and_save_pc(self, sample_id, args, is_canonical=False, is_test=False, verbose=False):
       
        import open3d as o3d #importing it here to avoid a bug

        joint_limits = self.model.jnt_range
        control_limits = self.model.actuator_ctrlrange
        joint_position = self.control_robot.control_to_non_collision_pose(
                                      control_limits, 
                                      joint_limits,
                                      is_canonical, 
                                      max_n_collisions=10,
                                      max_iter=100,
                                      robot_name=self.robot_name)
        if joint_position is None:
            if verbose:
                print(f"Failed to find a non-collision pose for sample {sample_id}. Skipping.")
            return
        #generate list of camera poses that are going to be used for this sample
        #since point clouds are also generated here, make sure to include sufficient camera angles
        camera_params = self.generate_camera_params() 
        images, depth_images, intrinsic_matrices, extrinsic_matrices = self.render_images(joint_position, 
                                                                                          camera_params,
                                                                                          render_depth=True,
                                                                                          lookat=[0, 0, 0])

        pcds = []
        for i in range(len(images)):
            color = o3d.geometry.Image(images[i])
            depth = o3d.geometry.Image(depth_images[i])

            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                images[i].shape[0],
                images[i].shape[1],
                intrinsic_matrices[i][0, 0],
                intrinsic_matrices[i][1, 1],
                intrinsic_matrices[i][0, 2],
                intrinsic_matrices[i][1, 2]
            )

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, convert_rgb_to_intensity=False, depth_scale=1.0
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic=pinhole_camera_intrinsic
            )
            pcd.transform(np.linalg.inv(extrinsic_matrices[i]))
            pcds.append(pcd)

        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds: #downsample point cloud from each viewpoint
            voxel_size = 0.0005
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            combined_pcd += pcd
        voxel_size = 0.0001
        while len(combined_pcd.points) > self.pcd_max_points: #globally downsample
            voxel_size *= 1.01
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)

        #file saving stuff
        directory_name = f"sample_{sample_id}" 
        if is_canonical:
            directory_name = "canonical_" + directory_name
        if is_test:
            directory_name = "test_" + directory_name
        unique_dir = os.path.join(self.save_dir, directory_name)

        # Use FileLock to ensure only one actor creates the directory
        lock_file = f"{unique_dir}.lock"
        with FileLock(lock_file):
            if not os.path.exists(unique_dir):
                os.makedirs(unique_dir, exist_ok=True)
                o3d.io.write_point_cloud(os.path.join(unique_dir, 'pc.ply'), combined_pcd)
                np.save(os.path.join(unique_dir, 'joint_positions.npy'), joint_position)

        # Save other files (these operations are safe to perform concurrently)
        for i, img in enumerate(images):
            img = Image.fromarray(img)
            img.save(os.path.join(unique_dir, f"image_{i}.jpg"))

        for i, img in enumerate(depth_images):
            np.save(os.path.join(unique_dir, f"depth_{i}.npy"), img)

        np.save(os.path.join(unique_dir, 'extrinsics.npy'), extrinsic_matrices)
        np.save(os.path.join(unique_dir, 'intrinsics.npy'), intrinsic_matrices)

    def render_images(self, joint_position, camera_params, render_depth=True, lookat=[0, 0, 0]):
        num_camera_params = camera_params.shape[0]
        images = []
        depth_images = []
        intrinsic_matrices = []
        extrinsic_matrices = []
        # lookat = self.look_at_end_effector()
        for j in range(num_camera_params):
            # self.random_base_placement()
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance = camera_params[j, 0]
            cam.azimuth = camera_params[j, 1]
            cam.elevation = camera_params[j, 2]
            cam.lookat = np.array(lookat)
            # print(camera_params[j,:])
            self.data.qpos[:] = joint_position
            mujoco.mj_step(self.model, self.data)

            self.renderer.update_scene(self.data, camera=cam)
            
            pixels = self.renderer.render()

            if render_depth:
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render()
                self.renderer.disable_depth_rendering()
            else:
                depth = None

            images.append(pixels)
            depth_images.append(depth)
            intrinsic_matrices.append(compute_camera_intrinsic_matrix(self.model, self.renderer, self.data))
            extrinsic_matrices.append(compute_camera_extrinsic_matrix(cam))

        return images, depth_images, intrinsic_matrices, extrinsic_matrices
    
    def generate_camera_params(self): #this can be implemented in other ways

        azimuth_range = np.linspace(-180, 180, 4)
        elevations = [-45, 45]
        radii = np.array([1.0, 2.0]) * args.camera_distance_factor
        camera_params = np.zeros((len(azimuth_range) * len(elevations) * len(radii), 3))

        azimuth_offset = np.random.uniform(-180, 180)
        elevation_offset = np.random.uniform(-25, 25)
        radius_offset = np.random.uniform(-0.5, 0.5) * args.camera_distance_factor

        for i, (azimuth, elevation, radius) in enumerate(itertools.product(azimuth_range, elevations, radii)):
            azimuth += azimuth_offset
            elevation += elevation_offset
            radius += radius_offset
            
            camera_params[i] = [radius, azimuth, elevation]

        return camera_params
    
    def look_at_end_effector(self):

        # Compute position adjustment to keep end-effector in frame
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
        
        # Initial simulation step to update kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial end-effector position
        initial_site_pos = self.data.site_xpos[site_id]
        
        # Adjust base position to center end-effector
        # self.model.body('pitch_link').pos[:] = -initial_site_pos

        return initial_site_pos

    def random_base_placement(self, offset=0.02):
        # Random rotation angles
        # angle_x = np.random.uniform(-np.pi/4, np.pi/4)  # x-axis rotation
        # angle_y = np.random.uniform(-np.pi/4, np.pi/4)  # y-axis rotation
        # angle_z = np.random.uniform(-np.pi/4, np.pi/4)  # z-axis rotation
        angle_x = 0  # x-axis rotation
        angle_y = 0  # y-axis rotation
        angle_z = 0  # z-axis rotation
        # print("angle_x: ", angle_x)
        # print("angle_y: ", angle_y)
        # print("angle_z: ", angle_z)
        # Compute quaternion for rotation
        def angle_to_quat(angle, axis):
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            quat = np.zeros(4)
            quat[0] = c
            quat[axis+1] = s
            return quat

        # Compute quaternion
        quat_x = angle_to_quat(angle_x, 0)
        quat_y = angle_to_quat(angle_y, 1)
        quat_z = angle_to_quat(angle_z, 2)

        # Combine rotations (quaternion multiplication)
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

        final_quat = quat_mul(quat_mul(quat_x, quat_y), quat_z)

        # Set base link orientation and position
        self.model.body('baselink').quat[:] = final_quat
        
        # Compute position adjustment to keep end-effector in frame
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
        
        # Initial simulation step to update kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial end-effector position
        initial_site_pos = self.data.site_xpos[site_id]
        
        # Adjust base position to center end-effector
        self.model.body('baselink').pos[:] = -initial_site_pos

        # ## add a noise in the base position
        # self.model.body('baselink').pos[0] += np.random.uniform(-offset, offset)
        # self.model.body('baselink').pos[1] += np.random.uniform(-offset, offset)
        # self.model.body('baselink').pos[2] += np.random.uniform(-offset, offset)

def generate_data(num_actors, num_samples, model_xml_dir, save_dir, args, is_canonical=False, is_test=False, verbose=False):

    actors = [MujocoActor.remote(actor_id, model_xml_dir, save_dir) for actor_id in range(num_actors)]

    tasks = []
    for i in range(num_samples):
        actor_index = i % num_actors
        task = actors[actor_index].generate_and_save_pc.remote(i, args, is_canonical, is_test, verbose)
        tasks.append(task)

    robot_name = os.path.basename(model_xml_dir)
    sample_type = 'canonical' if is_canonical else 'test' if is_test else 'normal'
    pbar = tqdm(total=num_samples, desc=f"Generating {sample_type} data for {robot_name}")
    start_time = time.time()

    while True:
     
        sample_type_prefix = "canonical_" if is_canonical else "test_" if is_test else ""
        num_files = len(glob.glob(os.path.join(save_dir, f"{sample_type_prefix}sample_*")))
        pbar.n = num_files
        pbar.refresh()

        if num_files >= num_samples:
            break

        time.sleep(1)

        elapsed_time = time.time() - start_time
        rate = num_files / elapsed_time
        pbar.set_description(f"Generating {sample_type} data for {robot_name} (Rate: {rate:.2f} files/sec)")

    pbar.close()
    
    # Clean up lock files
    clean_lock_files(save_dir)

def clean_lock_files(directory):
    lock_files = glob.glob(os.path.join(directory, "*.lock"))
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
        except OSError as e:
            print(f"Error deleting lock file {lock_file}: {e}")

if __name__ == "__main__":
    import time 
    import argparse 
    import shutil

    parser = argparse.ArgumentParser(description='Set model XML path and dataset name.')
    parser.add_argument('--model_xml_dir', type=str, default="mujoco_menagerie/universal_robots_ur5e", help='Path to the model XML file.')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset.')
    parser.add_argument('--num_canonical_samples', type=int, default=500, help='Number of canonical samples.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples.')
    parser.add_argument('--num_test', type=int, default=500, help='Number of test samples.')
    parser.add_argument('--num_actors', type=int, default=20, help='Number of actors.')
    parser.add_argument('--camera_distance_factor', type=float, default=0.15, help='Factor to scale the camera distance, change this depending on robot size.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()

    model_xml_dir = args.model_xml_dir   
    if not args.dataset_name:
        dataset_name = os.path.basename(model_xml_dir)
    else:
        dataset_name = args.dataset_name

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"./data/{dataset_name}"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True) 
    ray.init()

    num_actors = args.num_actors
    num_canonical_samples = args.num_canonical_samples
    num_samples = args.num_samples
    num_test = args.num_test

    if args.debug:
        num_samples = 100
        num_canonical_samples = 12
        num_test = 12
        num_actors = 1

    assert num_samples % num_actors == 0
    assert num_canonical_samples % num_actors == 0
    assert num_test % num_actors == 0
    
    generate_data(num_actors, 
                  num_canonical_samples, 
                  model_xml_dir,
                  save_dir, 
                  args=args,
                  is_canonical=True,
                  is_test=False, 
                  verbose=args.verbose)
    generate_data(num_actors, num_test, model_xml_dir, save_dir, args=args, is_test=True, is_canonical=False, verbose=args.verbose)
    generate_data(num_actors, num_samples, model_xml_dir, save_dir, args=args, is_test=False, is_canonical=False, verbose=args.verbose)

    # Final cleanup of any remaining lock files
    clean_lock_files(save_dir)
