# Depth Estimation with TensorFlow

This repository contains Python code and Jupyter Notebooks which are supposed to be able to train CNN which can estimate depth from a single input image.  
The source code is a modified and extended version of the code in this GitHub repository [siddinc/monocular_depth_estimation](https://github.com/siddinc/monocular_depth_estimation).

## TensorFlow installation on Windows native

Installation of TensorFlow 2.10 is as describe here: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip):

0. nvidia drivers (my gpu is RTX 3050 Ti Laptop)
1. anaconda
2. tf virtual environment in anaconda with python 3.9.18
3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
4. python -m pip install "tensorflow<2.11"
5. check if GPU is accesible: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
6. conda install jupyter
7. install with pip the rest of the packages for opencv, etc. when python gives you the module not found error

## Dataset

The used dataset is NYU Depth V2. It can be downloaded from kaggle: [https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) (4GB).
