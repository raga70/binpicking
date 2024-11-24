### How to get to running  condition :
- you need Pythonn 3.6  (I use an image of ubuntu 18)
- download  [models_zoo.zip](https://drive.google.com/file/d/1fbC0sGtVEUmAy7WPT_J-50IuIInMR9oO/view) and extract the modles that you want to use in ./models
- run `pip3 install .`    i have modified the dependencies to allow you to build the project in the current day 

### how to use :

- for now run ./examples/policy.py 
<br/>
<br/>
the params  are: 
     - model_name: Name of the GQ-CNN model to use.

    - depth_image_filename: Path to a depth image (float array in .npy format).

    - segmask_filename: Path to an object segmentation mask (binary image in .png format).

    - camera_intr_filename: Path to a camera intrinsics file (.intr file generated with BerkeleyAutomation’s perception package).

 Ofcourse we will create our own policy runner soon    