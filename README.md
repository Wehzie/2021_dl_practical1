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
