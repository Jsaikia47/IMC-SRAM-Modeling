# Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design
By Jyotishman Saikia, Bo Zhang, Jian Meng, Mingoo Seok and Jae-sun Seo.

Collaborative effort between Arizona State University  and Columbia University.



### Table of Contents:  

0.	Introduction
1.	Citation
2.	How to run
3.	Models

## Introduction

This repository contains the models described in the paper “Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design”. The work studies in-depth the effect of the parameters of a capacitive IMC SRAM design on network-level accuracy.

## Citation
To be added.

## How to run

Each of the model directories, include a subfolder ‘./models/’. The convolution and Fully connected layer with the bit-by-bit IMC implementation are described in this subfolder. The model layers description is listed in the same directory.


Run the code by running the shell script, ‘evaluate.sh’ added to each model. Default arguments are added and can be changed to user input. 

## Models

Modeled Resnet18 network for datasets CIFAR-10 and ImageNet. Corresponding Baseline accuracies are listed below.

Model Weight/Activation Precision   |    CIFAR-10   |     ImageNet
------------------------------------|---------------|---------------
1-bit |  88.98 % |   n/a
2-bit |  90.52 % |  62.83 %
4-bit |  91.47 % |  69.16 %
