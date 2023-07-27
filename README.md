# Workflow for Executing CNN Networks on Zynq Ultrascale+ with VITIS AI  🚀

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The rapid advancement of Artificial Intelligence has enabled remarkable achievements in various fields, such as image/video classification, semantic segmentation, object detection/tracking, and other applications applicable in embedded systems. However, the increasing size and complexity of deep learning models required to tackle large-scale image problems demand dedicated hardware acceleration on embedded devices for real-time processing with low latency.

## Overview 🔍

In this project, we present a detailed analysis of configuring, installing, and running a Convolutional Neural Network (CNN) on the Zynq Ultrascale+ board (ZCU102). This platform combines a multiprocessor system-on-chip (MPSoC) with an FPGA, offering hardware acceleration specifically tailored for efficient execution of neural networks, minimizing the computational load on the main processor.

The main goal of this study is to evaluate the **performance degradation when running a CNN model on the Deep Learning Processing Unit (DPU) of ZCU102 compared to running it on a Cloud infrastructure**. For this purpose, we have chosen to use Kaggle as the Cloud platform. Through a comparative and qualitative analysis, various relevant aspects will be considered to provide an assessment of the performance on the target board ZCU102.

![ZCU102](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/dc98d77c-8c22-4caf-9ea2-40a0cbe35b12)

## The Deep Learning Processor Unit (DPU) 💻

The Deep Learning Processor Unit (DPU) is a programmable and optimized processing unit specifically designed for implementing deep learning neural networks. Configurable using Xilinx's Vivado platform, the DPU accelerates operations of convolutional neural networks, including convolutions, pooling, and normalization. Its specialized instruction set maximizes neural network efficiency, reducing energy consumption and processing time. The DPU's hardware architecture utilizes a pipelined design, ensuring high parallelism and superior throughput compared to the CPU on the board. On-chip memory (OCM) minimizes external memory access, increasing overall system efficiency. The DPU is highly configurable and flexible, adapting to various applications and supporting widely used neural network architectures in Computer Vision, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, and more. The following sections will cover the configuration and complete installation of the DPU on the board.

![DPU](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/cbca3c51-95e0-4079-b991-380ae7344096)

## Traffic Sign Recognition System - Overview 🚦📸

This project aims to demonstrate the applicability of ZCU102 in a real-world context, supporting traffic sign recognition applications through a convolutional neural network. The implemented system is capable of recognizing signs such as turn indications, speed limits, no-overtaking zones, children pedestrian crossings, and other related symbols. The goal is to enable vehicles to take appropriate precautions in response to these signals.

We will develop a convolutional neural network model using the PyTorch library. To assess the performance, the model will be initially executed on the cloud via Kaggle and subsequently deployed on the ZCU102 board. Comparative and qualitative analysis will be conducted to identify any performance differences between execution on the ZCU102 Deep Learning Processing Unit (DPU) and the Cloud.

The dataset used is the Traffic Signs Classification, consisting of over 40 classes of traffic signs and more than 50,000 images. Its breadth and realism provide the model with a comprehensive understanding of traffic signs, enhancing its recognition and classification capabilities. This dataset is acquired from the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/).

![Traffic Signs](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/6030b36b-f1f0-4515-93be-617842074c00)

You can find the complete implementation of the model on **Kaggle** in the [GitHub repository](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/blob/main/traffic_sign_recognition_lenet_vgg16.ipynb), which contains the **notebook** for the project. This serve for save the necessary file **"model.pth"** that will be utilized for the AI Quantizer and AI Compiler

### Tools and Technologies Used 🛠️

- PyTorch
- Vivado
- Vitis AI
- Xilinx Deep Learning Processing Unit (DPU)
- PetaLinux
- Kaggle Cloud Platform

To experiment with the application's performance on the board, we build two convolutional networks for the traffic sign recognition problem, which is useful, for example, in the context of autonomous driving technology. We propose the LeNet model and the VGG16 model.

**LeNet**
![LeNet Model](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/2705e04d-d43f-4678-af20-baa3064ea98e)

**VGG16**
![VGG16 Model](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/8f195511-2c6a-434b-914e-845859ed4e3d)

### Configuration of Petalinux, DPU, and Vitis AI on Host and Zynq Ultrascale+ ZCU102 🔧

The configuration before running the support flow for model execution on the board is detailed in the [complete Italian 🇮🇹 document](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/blob/main/Configurazione%2C%20installazione%20ed%20esecuzione%20di%20una%20CNN%20sulla%20DPU%20della%20Zynq%20Ultrascale%2B_%20Valutazione%20delle%20prestazioni%20rispetto%20ad%20una%20GPU%20in%20Cloud%20-%20Embedded%20Systems%20wiki.pdf).

### Vitis AI Execution Flow with PyTorch for ZCU102 📊

Description of the Vitis AI execution flow with PyTorch for ZCU102: 

![Execution Flow](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/40d3f43d-0196-4dfa-b5e7-f76766305d9b)

After following the described process, the folder "target_zcu102" will be generated, containing the files to be uploaded to the board, as shown in the figure: 

![Folder Contents](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/51153cc1-d8cd-44fb-8117-1f03172082f4)

By transferring the folder correctly to ZCU102, the application can be run on the board using the following command:

```bash
python3 app_traffic_sign.py -m CNN_zcu102.xmodel  --threads (N) 
```

By executing all possible configuration combinations between the number of cores and threads, the results are as follows:

#### Performance Data of Respective Configurations and Executions 📈

| Platform | Throughput (fps) | Accuracy (%) |
|--------------|--------------|------------|
| Cloud (GPU P100) | LeNet: 2,685,901.64 | 0.9776 |
|                  | VGG16: 2,771,993.92 | 0.9856 |
| Host (Quantization 8-bit) | LeNet: N/A | 0.9739 |
|                            | VGG16: N/A | 0.9848 |
| ZCU102 DPU Single Core - 1 Thread | LeNet: 3,593.36 | 0.9772 |
|                                   | VGG16: 333.46 | 0.9839 |
| ZCU102 DPU Multi Core - 1 Thread | LeNet: 3,559.38 | 0.9772 |
|                                  | VGG16: 331.46 | 0.9839 |
| ZCU102 DPU Single Core - 4 Threads | LeNet: 5,529.12 | 0.9772 |
|                                    | VGG16: 352.58 | 0.9839 |
| ZCU102 DPU Multi Core - 4 Threads | LeNet: 5,631.97 | 0.9772 |
|                                  | VGG16: 486.66 | 0.9839 |

![Performance Graphs](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/58af9fbd-5222-4500-8e0d-0e6b523e07e7)
![Performance Graphs](https://github.com/LaErre9/Zynq_Ultrascale_Vitis_AI/assets/7995055/5378a05f-78ee-46e3-bd44-623f756c0b7b)

## Conclusions 🎯

Based on the results and observations made during the project, we can draw some conclusions about the use of Zynq UltraScale+ MPSoC (ZCU102) in Edge AI applications. The accuracy achieved by the model on the DPU is approximately equal to that obtained when the model is executed in the Cloud. However, the throughput (calculated as frame/sec) is significantly lower in the DPU compared to the Cloud solution.

This leads to a trade-off decision since this application is targeted for use on board an autonomous driving vehicle. Both high accuracy and adequate throughput are crucial, as shown in the case of the LeNet model in this study. Indeed, in the case of the multi-core DPU with 4 threads, the system is capable of performing traffic sign recognition in 0.17756 ms, without considering additional constraints such as the distance at which the model can perform recognition and others.

In general, despite the quantization during the Vitis-AI workflow, the model retains a good level of accuracy in its predictions. This is a significant advantage when the goal of the Edge AI application is to be as accurate as possible without major time constraints.

However, the main limitation of this approach lies in the technological constraints imposed by Zynq UltraScale+; even with the integration of a DPU that accelerates convolution operations for neural networks, the throughput is not yet comparable to that of a Cloud solution.

### Project developed for demonstrative and educational purposes ✅

© 2023 - Project developed for the Embedded Systems exam at the University of Naples, Federico II. Created exclusively for demonstrative and educational purposes.
*Authors*: **Antonio Romano** - **Giuseppe Riccio**

### References 📚

- [Vitis™ AI User Guide](https://docs.xilinx.com/r/2.0-English/ug1414-vitis-ai/Vitis-AI-Overview)
- [Documentation DPU for Zynq UltraScale+ MPSoC](https://docs.xilinx.com/r/en-US/pg338-dpu?tocId=3xsG16y_QFTWvAJKHbisEw)
- [Vitis™ AI Library User Guide](https://docs.xilinx.com/r/2.0-English/ug1354-xilinx-ai-sdk/Introduction)
- [PetaLinux Tools Documentation Reference Guide](https://docs.xilinx.com/r/2021.1-English/ug1144-petalinux-tools-reference-guide/Overview)

These documents were used as reference sources for the project, providing crucial information regarding the use and configuration of Xilinx resources and related development tools.
