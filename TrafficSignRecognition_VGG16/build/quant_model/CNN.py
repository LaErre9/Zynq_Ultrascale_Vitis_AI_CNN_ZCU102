# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[0]/input.2
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[1]/654
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[2]/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[3]/input.4
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[4]/688
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[5]/input.5
        self.module_7 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[6]/input.6
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[7]/722
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[8]/input.7
        self.module_10 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[9]/input.8
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[10]/756
        self.module_12 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[11]/input.9
        self.module_13 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #CNN::CNN/Sequential[model]/Conv2d[12]/input.10
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[13]/790
        self.module_15 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN::CNN/Sequential[model]/MaxPool2d[14]/804
        self.module_16 = py_nndct.nn.Module('flatten') #CNN::CNN/Sequential[model]/Flatten[15]/input.11
        self.module_17 = py_nndct.nn.Linear(in_features=512, out_features=4096, bias=True) #CNN::CNN/Sequential[model]/Linear[16]/input.12
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[17]/input.13
        self.module_19 = py_nndct.nn.Linear(in_features=4096, out_features=4096, bias=True) #CNN::CNN/Sequential[model]/Linear[18]/input.14
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #CNN::CNN/Sequential[model]/ReLU[19]/input
        self.module_21 = py_nndct.nn.Linear(in_features=4096, out_features=43, bias=True) #CNN::CNN/Sequential[model]/Linear[20]/821

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        output_module_0 = self.module_14(output_module_0)
        output_module_0 = self.module_15(output_module_0)
        output_module_0 = self.module_16(input=output_module_0, start_dim=1, end_dim=3)
        output_module_0 = self.module_17(output_module_0)
        output_module_0 = self.module_18(output_module_0)
        output_module_0 = self.module_19(output_module_0)
        output_module_0 = self.module_20(output_module_0)
        output_module_0 = self.module_21(output_module_0)
        return output_module_0
