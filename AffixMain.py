import os
import json
import glob
import time
from examples.policy import run_policy
import concurrent.futures

def process_files(parallel=False, gripper_mode=None):
    npy_dir = "AffixData/npy-pointCloud"
    segmask_dir = "AffixData/segmask"
    results_dir = "AffixData/Results"
    config_dir = "AffixConfig"
    os.makedirs(results_dir, exist_ok=True)

    def run_single_policy(model_name, fully_conv, depth_image, segmask=None, config_filename=None):
        start_time = time.time()
        result = run_policy(model_name=model_name, fully_conv=fully_conv,
                            depth_image=depth_image, segmask=segmask,
                            config_filename=config_filename)
        print(f"Calculating Grasp Point for {model_name} took {time.time() - start_time:.2f}s")
        return result

    for npy_file in glob.glob(os.path.join(npy_dir, "*.npy")):
        file_name = os.path.basename(npy_file)
        base_name = os.path.splitext(file_name)[0]
        segmask_file = os.path.join(segmask_dir, f"{base_name}.png")
        has_segmask = os.path.exists(segmask_file)

        suc_result = pj_result = None

        if gripper_mode in [None, "suction"]:
            model = f"{'FC-' if has_segmask else ''}GQCNN-4.0-SUCTION"
            config = f"dex-net_4.0_{'fc_' if has_segmask else ''}suction.yaml"
            suc_result = run_single_policy(model, has_segmask, npy_file,
                                           segmask_file if has_segmask else None,
                                           os.path.join(config_dir, config))

        if gripper_mode in [None, "pj"]:
            model = f"{'FC-' if has_segmask else ''}GQCNN-4.0-PJ"
            config = f"dex-net_4.0_{'fc_' if has_segmask else ''}pj.yaml"
            if parallel and gripper_mode is None:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    pj_future = executor.submit(run_single_policy, model, has_segmask, npy_file,
                                                segmask_file if has_segmask else None,
                                                os.path.join(config_dir, config))
                    pj_result = pj_future.result()
            else:
                pj_result = run_single_policy(model, has_segmask, npy_file,
                                              segmask_file if has_segmask else None,
                                              os.path.join(config_dir, config))

        result = {}
        if suc_result:
            result["suction"] = vars(suc_result)
        if pj_result:
            result["pj"] = vars(pj_result)

        if suc_result and pj_result:
            result["bestGripperToUse"] = "suction" if suc_result.certainty > pj_result.certainty else "pj"

        output_file = os.path.join(results_dir, f"{base_name}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run PJ and SUC policies in parallel")
    parser.add_argument("--gripper", choices=["suction", "pj"], help="Optional Flag Run only specified gripper type")
    args = parser.parse_args()
    process_files(args.parallel, args.gripper)