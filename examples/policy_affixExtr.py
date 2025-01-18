import argparse
import json
import os
import time
import numpy as np
from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
from scipy.spatial.transform import Rotation as R

logger = Logger.get_logger("examples/policy.py")

class ResultObj:
    def __init__(self, position, orientation, certainty):
        self.position = position
        self.orientation = orientation
        self.certainty = certainty

    def __repr__(self):
        return (f"Result(position={self.position}, "
                f"orientation={self.orientation}, "
                f"certainty={self.certainty})")


def grasp_to_pose(grasp):
    # Extract position (center in 2D and depth)
    center = grasp.center  # 2D point (x, y)
    depth = grasp.depth  # depth (z)

    # Position is in 3D space (center + depth as z-coordinate)
    position = np.array([center[0], center[1], depth])

    # Extract the orientation (approach_axis and approach_angle for rotation)
    axis = grasp.approach_axis  # 3D vector for the axis
    angle = grasp.approach_angle  # Angle for rotation

    # Convert axis-angle to quaternion using scipy
    rotation = R.from_rotvec(axis * angle)  # axis * angle gives the rotation vector
    quaternion = rotation.as_quat()  # Get quaternion [x, y, z, w]

    # Return position and quaternion as a pose
    pose = {
        'position': position.tolist(),  # Position in 3D space
        'orientation': quaternion.tolist(),  # Quaternion for orientation
        'frame': grasp.frame  # The frame associated with the grasp
    }

   
    return pose

def create_result_from_grasp(action_object):
    pose= grasp_to_pose(action_object.grasp)
    position = {
        "x": pose['position'][0],
        "y": pose['position'][1],
        "z": pose['position'][2],
    }
    orientation = {
        "x": pose['orientation'][0],
        "y": pose['orientation'][1],
        "z": pose['orientation'][2],
        "w": pose['orientation'][3],
    }
    return ResultObj(position=position, orientation=orientation, certainty=action_object.q_value)

def run_policy(model_name=None,
               depth_image=None,
               segmask=None,
               config_filename=None,
               camera_intr="./AffixConfig/CAMERAcfgPhoxi.intr",
               model_dir=None,
               fully_conv=True):

    # Validate inputs
    assert not (fully_conv and depth_image is not None and segmask is None), \
        "Fully-Convolutional policy expects a segmask."

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_path = os.path.join(model_dir, model_name)

    # Get configs
    model_config = json.load(open(os.path.join(model_path, "config.json"), "r"))
    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        gripper_mode = get_gripper_mode(input_data_mode)

    # Read config
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Update model path
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                policy_config["metric"]["gqcnn_model"])

    # Setup sensor and read images
    camera_intr = CameraIntrinsics.load(camera_intr)
    depth_data = np.load(depth_image)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Process segmask
    segmask = BinaryImage.open(segmask) if segmask else None
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    segmask = segmask.mask_binary(valid_px_mask) if segmask else valid_px_mask

    # Create state
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # Update policy config for fully conv
    if fully_conv:
        policy_config["metric"]["fully_conv_gqcnn_config"]["im_height"] = depth_im.shape[0]
        policy_config["metric"]["fully_conv_gqcnn_config"]["im_width"] = depth_im.shape[1]

    # Initialize and run policy
    policy = init_policy(fully_conv, policy_config)
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))

    # Save visualization
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config["vis"]["vmin"],
                   vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title(f"Planned grasp at depth {action.grasp.depth:.3f}m with Q={action.q_value:.3f}")

        # Generate output filename
        npy_name = os.path.splitext(os.path.basename(depth_image))[0]
        model_type = "pj" if "PJ" in model_name else "suc"
        output_path = os.path.join("..", "AffixOutput", f"{npy_name}_{model_type}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vis.savefig(output_path)

    return create_result_from_grasp(action)

def get_gripper_mode(input_data_mode):
    mode_map = {
        "tf_image": GripperMode.LEGACY_PARALLEL_JAW,
        "tf_image_suction": GripperMode.LEGACY_SUCTION,
        "suction": GripperMode.SUCTION,
        "multi_suction": GripperMode.MULTI_SUCTION,
        "parallel_jaw": GripperMode.PARALLEL_JAW
    }
    if input_data_mode not in mode_map:
        raise ValueError(f"Input data mode {input_data_mode} not supported!")
    return mode_map[input_data_mode]

def init_policy(fully_conv, policy_config):
    if fully_conv:
        if policy_config["type"] == "fully_conv_suction":
            return FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config["type"] == "fully_conv_pj":
            return FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
        raise ValueError(f"Invalid fully-convolutional policy type: {policy_config['type']}")

    policy_type = policy_config.get("type", "cem")
    if policy_type == "ranking":
        return RobustGraspingPolicy(policy_config)
    elif policy_type == "cem":
        return CrossEntropyRobustGraspingPolicy(policy_config)
    raise ValueError(f"Invalid policy type: {policy_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grasping policy on an example image")
    parser.add_argument("model_name", type=str, help="name of a trained model to run")
    parser.add_argument("--depth_image", type=str, help="path to a test depth image stored as a .npy file")
    parser.add_argument("--segmask", type=str, help="path to an optional segmask to use")
    parser.add_argument("--camera_intr", type=str, help="path to the camera intrinsics")
    parser.add_argument("--model_dir", type=str, help="path to the folder in which the model is stored")
    parser.add_argument("--config_filename", type=str, help="path to configuration file to use")
    parser.add_argument("--fully_conv", action="store_true",
                        help="run Fully-Convolutional GQ-CNN policy instead of standard GQ-CNN policy")

    args = parser.parse_args()
    result = run_policy(
        model_name=args.model_name,
        depth_image=args.depth_image,
        segmask=args.segmask,
        config_filename=args.config_filename,
        camera_intr=args.camera_intr,
        model_dir=args.model_dir,
        fully_conv=args.fully_conv
    )
    print(result)