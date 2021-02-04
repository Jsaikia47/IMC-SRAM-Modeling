# Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design
By Jyotishman Saikia, Shihui Yin, Sai Kiran Cherupally, Bo Zhang, Jian Meng, Mingoo Seok and Jae-sun Seo.

Collaborative effort between Arizona State University  and Columbia University.



### Table of Contents:  

0.	Introduction
1.	Citation
2.	How to run
3.	Models

## Introduction

This repository contains the models described in the paper “Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design”. The work studies in-depth the effect of the parameters of a capacitive IMC SRAM design on network-level accuracy.

## Citation
If you use these models in your research, please cite:

Jyotishman Saikia, Shihui Yin, Sai Kiran Cherupally, Bo Zhang, Jian Meng, Mingoo Seok, and Jae-sun Seo, “Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design,” IEEE Design, Automation & Test in Europe (DATE), February 2021.


@inproceedings{saikia_date21,
  title={{Modeling and Optimization of SRAM-based In-Memory Computing Hardware Design}},
  author={Saikia, Jyotishman and Yin, Shihui and Cherupally, Sai Kiran and Zhang, Bo and Meng, Jian and Seok, Mingoo and Seo, Jae-sun},
  booktitle={Design, Automation, and Test in Europe (DATE)},
  year={2021}
}

## How to run

Each of the model directories, include a subfolder ‘./models/’. The convolution and Fully connected layer with the bit-by-bit IMC implementation are described in this subfolder. The model layers description is listed in the same directory.


Run the code by running the shell script, ‘evaluate.sh’ added to each model directory. Default arguments are added for ease and can be changed to user input. 

## Models

Modeled Resnet18 network for datasets CIFAR-10 and ImageNet. Corresponding baseline accuracies are listed below.

Model Weight/Activation Precision   |    CIFAR-10   |     ImageNet
------------------------------------|---------------|---------------
1-bit |  88.98 % |   n/a
2-bit |  90.52 % |  62.83 %
4-bit |  91.47 % |  69.16 %
