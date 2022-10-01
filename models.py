import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as pair
import keras
from keras import Sequential, initializers, layers

class FCNet(nn.Module):
    def __init__(self, args):
        units = [64, 128, 256]

        super(FCNet, self).__init__()
        self.d = args.dropout
        self.init = args.init
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(256, units[0])
        self.linear2 = nn.Linear(units[0], units[1])
        self.linear3 = nn.Linear(units[1], units[2])
        self.linear4 = nn.Linear(units[2],4096)
        self.linear5 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(p=args.dropout)

        if self.init != 4:
            layers = {str(i+1):x for i,x in enumerate([self.linear1, self.linear2, self.linear3, self.linear4])}

            for i in range(1, len(layers)+1):
                torch.nn.init.uniform_(layers[str(i)].weight, *self.init_range(layers[str(i)].weight.shape[1]))
                torch.nn.init.constant_(layers[str(i)].bias, 1e-1)
            
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        if self.d > 0:
            x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        if self.d > 0:
            x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu(x)
        if self.d > 0:
            x = self.dropout(x)
        
        x = self.linear4(x)
        x = self.tanh(x)
        if self.d > 0:
            x = self.dropout(x)

        x = self.linear5(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def init_range(self, shape):
        values = [(-1/math.sqrt(shape), 1/math.sqrt(shape)), (2e-1, 8e-1), (-3e-2, 3e-2)]
        ranges = {str(i+1):x for i,x in enumerate(values)}
        return ranges[str(self.init)]
        

class ConvNet(nn.Module):
    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.init = args.init
        self.padding = pair(0)
        self.dilation = pair(1)
        self.stride = pair(1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 10)

        if self.init != 4:
            layers = {str(i+1):x for i,x in enumerate([self.conv1, self.conv2, self.conv3])}
            dims = [(16,16)]

            for i in range(1, len(layers)+1):
                dims.append(pair(self.calc_output_dim(dims[i-1][0], layers[str(i)].kernel_size)))
            for i in range(1, len(layers)+1):
                torch.nn.init.uniform_(
                        layers[str(i)].weight, 
                        *self.init_range(
                            layers[str(i)].weight.shape[0] * pow(layers[str(i)].weight.shape[-1], 2), 
                            layers[str(i)].kernel_size[0]))

                torch.nn.init.constant_(layers[str(i)].bias, 1e-1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
       
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def init_range(self, shape, k_size):
        values = [(k_size * -1/math.sqrt(shape), k_size * 1/math.sqrt(shape)), (-2, 2), (-4e-2,7e-2)]
        ranges = {str(i+1):x for i,x in enumerate(values)}
        return ranges[str(self.init)]

    def calc_output_dim(self, input_size, k_size):
        return int(F.math.floor(
                    ((input_size + 2 * self.padding[0] - self.dilation[0] * (k_size[0] - 1) - 1) 
                    / self.stride[0]) + 1))

def LCNet(args):
    def init_range(init, shape, k_size=3):
        values = [(k_size * -1/math.sqrt(shape), k_size * 1/math.sqrt(shape)), (-2,2), (-3e-2,3e-2)]
        ranges = {str(i+1):x for i,x in enumerate(values)}
        return ranges[str(init)]

    def calc_output_dim(input_size, k_size=3, padding=0, dilation=1, stride=1):
        return int(F.math.floor(
                    ((input_size + 2 * padding - dilation * (k_size - 1) - 1) 
                    / stride) + 1))

    if args.init != 4:
        dims = [(16,16)]
        for i in range(0, 3):
            dims.append(pair(calc_output_dim(dims[i][0])))
        
        k_initializer = []
        for i in range(0, len(dims)):
            k_initializer.append(keras.initializers.RandomUniform(*init_range(args.init, pow(dims[i][0],2) * 9 // 8)))
    else:
        k_initializer = [initializers.GlorotUniform()] * 3  
    
    b_initializer = initializers.Constant(1e-1)
    
    model = Sequential([
        layers.Input(shape=(16,16,1)),
        layers.LocallyConnected2D(8, (3,3),
                activation='relu',
                kernel_initializer=k_initializer[0],
                bias_initializer=b_initializer,
                implementation=2),
        layers.LocallyConnected2D(8, (3,3),
                activation='relu',
                kernel_initializer=k_initializer[1],
                bias_initializer=b_initializer,
                implementation=2),
        layers.LocallyConnected2D(8, (3,3),
                activation='tanh',
                kernel_initializer=k_initializer[2],
                bias_initializer=b_initializer,
                implementation=2),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    
    return model

