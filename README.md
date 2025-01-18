Bin Picking System - Generate Pick(Grasp) Position   (updated DEX-NET)

This system leverages an updated and modified version of **Dex-Net** to perform object segmentation and generate grasp positions from `.ply` files. It supports both **parallel-jaw grippers** and **vacuum grippers**, providing recommendations on which gripper is better suited for each object. Additionally, the system can optionally label the objects.

Your **vision system** should output new 3D scans of the bin as `.ply` files to the watch folder located at `AffixData/ply-inputs/`. These files are then processed, and the results are saved as individual JSON files for each object in the `AffixData/Results/` directory. The JSON files include detailed information, such as segmentation, grasp positions, and gripper recommendations.

An example connector script, `AffixMech-VisionConnector.py`, is included to demonstrate how to integrate the generated data with a **Mech-Vision** application. You can use this script as a reference



## 1. Installation:


prerqusits: 

- create a VENV for python 3.6.9 ,
- have at least 10gb of RAM,
- the use Of a GPU with at least 10gb of VRAM is recommended  , (CPU processing is also available but the calculation of graspingPoint for an Object takes ~5sec)  (have your cuda drivers installed, the solution will automatically detect that you have a GPU and use it)
- If you have issues with Memory and DexNet you can run `PreapreRadosEnv.sh`  witch will allow memory overcommit on your system
- the Solution was designed and tested on Ubuntu 18.04  (you can use WSL)

1. Setup Mechmind to output  .ply to `AffixData/ply-inputs/` 
2. install python 3.6.9  and run setup.py

For Labeling of the Data(OPTIONAL):

1. install python 3.8 for yolo and `pip install ultralytics`  in python 3.8 
2.  go to `AffixSam2/` and `pip install`  with python 3.8
- Run AffixMain.py with `--label`

## 2. Configuration:

**1. Calibrate 3D segmentator for your setup :**

 run  the autoCalibrator: `python3 AffixObjectSegmentator.py --calibrate` 

follow the steps outlined in the calibration process to chose the best values

modify the default values for “eps",  “min_samples", “min_certainty”, in AfficObjectSegmentator.py

if you want to run in Debug Mode you can run the AfficObjectSegmentator.py with  `--visualizeSegmentedObjs`  

**2. modify TF of the camera :**

edit `AffixConfig/CAMERAcfgPhoxi.intr`  to reflect your enviorment

 

3. additional configuration(OPTIONAL):

you can fine tune  the config of the DEX-NET models  by editing `AffixConfig/dex-net_4.0_fc_pj.yaml`  and `AffixConfig/dex-net_4.0_fc_suction.yaml`

## 3. Testing out different components :

1. provide at least one .ply file in `AffixData/ply-inputs/`

1. test 3D segmentation:

```bash
python3 AffixObjectSegmentator.py --visualizeSegmentedObjs
```

1. test Dex-Net:    (AffixData/npy-pointCloud/fullCloud_object_0.npy may not exist)

```bash
python3 examples/ploicy_affixExtr.py FC-GQCNN-4.0-SUCTION --fully_conv --depth_image AffixData/npy-pointCloud/fullCloud_object_0.npy --segmask AffixData/npy-pointCloud/fullCloud_object_0_mask.png --config_filename cfg/examples/replication/dex-net_4.0_fc_suction.yaml --camera_intr data/calib/phoxi/phoxi.intr
```

```bash
python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image AffixData/npy-pointCloud/fullCloud_object_0.npy --segmask AffixData/npy-pointCloud/fullCloud_object_0_mask.png --config_filename cfg/examples/replication/dex-net_4.0_fc_pj.yaml --camera_intr data/calib/phoxi/phoxi.intr
```

1. Test  SAM2 And Yolo (for Labeling)

located in `AffixSam2/SAM_Centers.py` ,  `AffixYoloClassification/main.py`

## Run The Solution:

Run AffixMain.py in the background

```bash
python3 AffixMain.py  {args}
```

args:

```
  --parallel               Run PJ and SUC policies in parallel
  
  --gripper {suction,pj}   Run only the specified gripper type
  
  --label LABEL            Optionally label each object with Yolo

```

the system will output Json files per each object in `AffixData/Results`

connecting with Mech-Vison

an **example** connector is created called `AffixMech-VisionConnector.py`

its `process` method returns a :

```
tuple:
            - np.array: PoseList containing all extracted poses.
            - list: A boolean list where True indicates Suction, and False indicates PJ.
            - list: A list of labels, where None indicates no label present.
```

adapted to machined data types , modify the return of this method to suite your setup

(the tuple is not acceptable by mech-vision each dataType inside it is !!! modification required to adapt to your setup!!!)
