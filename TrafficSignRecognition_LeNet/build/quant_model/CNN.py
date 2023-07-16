# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[0]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[2]/507
        self.module_4 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[3]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[4]/input.5
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[6]/546
        self.module_8 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[7]/input.7
        self.module_9 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[8]/input.8
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[10]/585
        self.module_12 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[11]/599
        self.module_13 = py_nndct.nn.Module('flatten') #CNN::CNN/Sequential[model]/Flatten[12]/input.10
        self.module_14 = py_nndct.nn.Linear(in_features=576, out_features=256, bias=True) #CNN::CNN/Sequential[model]/Linear[13]/input.11
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[14]/input
        self.module_16 = py_nndct.nn.Linear(in_features=256, out_features=43, bias=True) #CNN::CNN/Sequential[model]/Linear[15]/611

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(input=output_module_0, start_dim=1, end_dim=3)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(output_module_0)
        return output_module_0
