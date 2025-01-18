import os
import json
import glob
import subprocess
import time
from examples.policy_affixExtr import run_policy
import concurrent.futures
import multiprocessing

from AffixObjectSegmentator import main as affixObjectSegmentator

def classify_object_with_subprocess(npy_file):

    try:
        # Run the Python 3.8 wrapper script to segment  the objects with SAM2  needs python 3.8
        claify_result = subprocess.run(
            ["python3.8", "AffixSam2/SAM_Centers.py"],
            text=True,
            capture_output=True,
            check=True
        )
        # Parse the JSON output
        output = json.loads(claify_result.stdout)
        print("RGB Segmentation result: ", output)
    except subprocess.CalledProcessError as e:
        print(f"Error during RGB segmentation: {e.stderr}")
        return ""

    try:
        # Run the Python 3.8 wrapper script to classify the object yolo needs python 3.8
        claify_result = subprocess.run(
            ["python3.8", "AffixYoloClassification/classify_wrapper.py", npy_file],
            text=True,
            capture_output=True,
            check=True
        )
        # Parse the JSON output
        output = json.loads(claify_result.stdout)
        return output.get("label", "")
    except subprocess.CalledProcessError as e:
        print(f"Error during classification: {e.stderr}")
        return ""




ply_dir = "AffixData/ply-inputs"

npy_dir = "AffixData/npy-pointCloud"
segmask_dir = "AffixData/npy-pointCloud"
results_dir = "AffixData/Results"
config_dir = "AffixConfig"


def run_policy_in_process(queue, *args, **kwargs):
    """Runs the policy and places the result in the queue."""
    try:
        result = run_policy(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)


def run_single_policy_with_timeout(*args, **kwargs):
    """Runs the policy with a 15-second timeout, ensuring proper process isolation."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_policy_in_process, args=(queue, *args), kwargs=kwargs)
    process.start()

    process.join(timeout=15)
    if process.is_alive():
        process.terminate()  # Force kill the process if it exceeds the timeout
        process.join()
        print(f"\033[91mError: Calculating Grasp Point took too long (exceeded 15 seconds)\033[0m")
        return None

    # Retrieve the result from the queue
    if not queue.empty():
        result = queue.get()
        if isinstance(result, Exception):
            print(f"\033[91mError: {result}\033[0m")
            return None
        return result
    else:
        print(f"\033[91mError: No result returned by run_policy\033[0m")
        return None


def process_npy_files(parallel=False, gripper_mode=None, label=False):
    os.makedirs(results_dir, exist_ok=True)

    

    for npy_file in glob.glob(os.path.join(npy_dir, "*.npy")):
        print("gripper: " + gripper_mode)
        file_name = os.path.basename(npy_file)
        base_name = os.path.splitext(file_name)[0]
        segmask_file = os.path.join(segmask_dir, f"{base_name}_mask.png")
        has_segmask = os.path.exists(segmask_file)
        classified_label = ""

        if label:
            classified_label = classify_object_with_subprocess(npy_file)  # Make the call via subprocess with Python 3.8

        suc_result = pj_result = None

        # Parallel Execution: Both Suction and PJ
        if parallel and gripper_mode is None:
            model_suc = f"{'FC-' if has_segmask else ''}GQCNN-4.0-SUCTION"
            config_suc = f"dex-net_4.0_{'fc_' if has_segmask else ''}suction.yaml"

            model_pj = f"{'FC-' if has_segmask else ''}GQCNN-4.0-PJ"
            config_pj = f"dex-net_4.0_{'fc_' if has_segmask else ''}pj.yaml"

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    "suction": executor.submit(
                        run_single_policy_with_timeout,
                        model_name=model_suc,
                        fully_conv=has_segmask,
                        depth_image=npy_file,
                        segmask=segmask_file if has_segmask else None,
                        config_filename=os.path.join(config_dir, config_suc)
                    ),
                    "pj": executor.submit(
                        run_single_policy_with_timeout,
                        model_name=model_pj,
                        fully_conv=has_segmask,
                        depth_image=npy_file,
                        segmask=segmask_file if has_segmask else None,
                        config_filename=os.path.join(config_dir, config_pj)
                    ),
                }
                suc_result = futures["suction"].result()
                pj_result = futures["pj"].result()

        # Non-Parallel Execution: Suction or PJ based on gripper_mode
        else:
            if gripper_mode in [None, "suction"]:
                model = f"{'FC-' if has_segmask else ''}GQCNN-4.0-SUCTION"
                config = f"dex-net_4.0_{'fc_' if has_segmask else ''}suction.yaml"
                suc_result = run_single_policy_with_timeout(
                    model_name=model,
                    fully_conv=has_segmask,
                    depth_image=npy_file,
                    segmask=segmask_file if has_segmask else None,
                    config_filename=os.path.join(config_dir, config)
                )

            if gripper_mode in [None, "pj"]:
                model = f"{'FC-' if has_segmask else ''}GQCNN-4.0-PJ"
                config = f"dex-net_4.0_{'fc_' if has_segmask else ''}pj.yaml"
                pj_result = run_single_policy_with_timeout(
                    model_name=model,
                    fully_conv=has_segmask,
                    depth_image=npy_file,
                    segmask=segmask_file if has_segmask else None,
                    config_filename=os.path.join(config_dir, config)
                )

        # Skip output generation if both results are None
        if suc_result is None and pj_result is None:
            print(f"\033[91mSkipping {npy_file} due to timeout or error\033[0m")
            continue

        result = {}
        if label:
            result["label"] = classified_label
        if suc_result:
            result["suction"] = vars(suc_result)
        if pj_result:
            result["pj"] = vars(pj_result)

        if suc_result and pj_result:
            result["bestGripperToUse"] = "suction" if suc_result.certainty > pj_result.certainty else "pj"

        output_file = os.path.join(results_dir, f"{base_name}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\033[92m{npy_file} processed successfully\033[0m")


def mainLoop(aParallel=False, aGripper=None, aLabel=False):
    def cleanOldFiles():
        """Remove all files in the npy directory and register existing .ply files as processed."""
        for npy_file in glob.glob(os.path.join(npy_dir, "*.npy")):
            os.remove(npy_file)
            print(f"Deleted old file: {npy_file}")
    
        return set(glob.glob(os.path.join(ply_dir, "*.ply")))


    processed_files = cleanOldFiles()
    print("gripper: " + aGripper)
    while True:
        # Monitor for new .ply files
        for ply_file in glob.glob(os.path.join(ply_dir, "*.ply")):
            if ply_file not in processed_files:
                processed_files.add(ply_file)
                # Delete all .npy files in the npy directory
                for npy_file in glob.glob(os.path.join(npy_dir, "*.npy")):
                    os.remove(npy_file)
                    print(f"Deleted: {npy_file}")
                print(f"\033[92m{ply_file} found. Starting Processing...\033[0m")
                affixObjectSegmentator()
                process_npy_files(aParallel, aGripper, aLabel)
        print(f"\033[93mWaiting for new .ply files in ./AffixData/ply-inputs/ ..... \033[0m")
        time.sleep(1)  # Avoid constant polling







if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process NPY files with optional parallelism and gripper specification.")
    parser.add_argument("--parallel", action="store_true", help="Run PJ and SUC policies in parallel")
    parser.add_argument("--gripper", choices=["suction", "pj"], help="Run only the specified gripper type")
    parser.add_argument("--label", type=str, help="Optionally label each object with Yolo")
    args = parser.parse_args()

    # Validation: Ensure --parallel and --gripper are mutually exclusive
    if args.parallel and args.gripper:
        print("\033[91mError: --parallel and --gripper cannot be used together.\033[0m", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

    # Validation: Ensure at least one of --parallel or --gripper is specified
    if not args.parallel and not args.gripper:
        print("\033[91mError: You must specify either --parallel or --gripper.\033[0m", file=sys.stderr)
        sys.exit(1)  # Exit with an error code

    # Print the selected mode for clarity
    if args.gripper:
        print(f"Running in single gripper mode: {args.gripper}")
    elif args.parallel:
        print("Running in parallel mode (both suction and PJ policies).")

    # Call the main loop
    mainLoop(args.parallel, args.gripper, args.label)

   # process_npy_files(args.parallel, args.gripper, args.label)   #FOR DEBUGGING