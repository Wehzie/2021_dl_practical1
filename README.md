# 2021 Deep Learning Practical 1

## Dataset

The dataset is the fashion-mnist provided by Zalando research.
The data contains 60,000 train and 10,000 test images.
Images belong to 10 classes.
The dimensionality of the images is 28x28; the images are grayscale.

## Experiments

Initial parameters

	optimizer: adam
	pooling: AVG
	weights: imagenet (pretrained, transfer learning)
	loss function: sparse categorical cross-entropy
	epochs: 5

1. MobileNetV2
2. MobileNet
3. DenseNet121
4. DenseNet169
   		
	--> Choose 1st and 2nd winner

5. 1st, MAX pooling
6. 2nd, MAX pooling

	--> Choose 1st winner

7. 1st, optimizer SDG
8. 1st, optimizer RMSprop

## Report

A max. 3 page report is found on Overleaf https://www.overleaf.com/project/601c07fdbef41b6d1745fe18

## Requirements

### Ubuntu

Install Tensorflow 

	# with NVIDIA GPU
	pip3 install tensorflow

	# without NVIDIA GPU
	pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl/cpu/tensorflow_cpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl

Install Keras
	pip3 install keras

Verify installation
	
	python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

Install NVIDIA CUDA

	# Add NVIDIA package repositories
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
	sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
	sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
	sudo apt-get update

	wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

	sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
	sudo apt-get update

	# Unresolved dependencies on Ubuntu 20.10
	sudo apt install libnvidia-gl-450 libnvidia-compute-450 libnvidia-decode-450 libnvidia-encode-450 libnvidia-ifr1-450 libnvidia-fbc1-450

	# Install NVIDIA driver
	sudo apt-get install --no-install-recommends nvidia-driver-450
	# Reboot. Check that GPUs are visible using the command: nvidia-smi

	wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
	sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
	sudo apt-get update

	# Install development and runtime libraries (~4GB)
	sudo apt-get install --no-install-recommends \
		cuda-11-0 \
		libcudnn8=8.0.4.30-1+cuda11.0  \
		libcudnn8-dev=8.0.4.30-1+cuda11.0


	# Install TensorRT. Requires that libcudnn8 is installed above.
	sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
		libnvinfer-dev=7.1.3-1+cuda11.0 \
		libnvinfer-plugin7=7.1.3-1+cuda11.0

