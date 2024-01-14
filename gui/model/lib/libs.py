import torch
import torch.nn as nn

#vgg
class encoder3(nn.Module):
    def __init__(self):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)
        
        self.pool2 = nn.MaxPool2d(2, 2)
            
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace = True)
            
    def forward(self, input):
        output = self.conv1(input)
        output = self.pad1(output)
        output = self.conv2(output)
        output = self.relu1(output)
        output = self.pad2(output)
        output = self.conv3(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.pad3(output)
        output = self.conv4(output)
        output = self.relu3(output)
        output = self.pad4(output)
        output = self.conv5(output)
        output = self.relu4(output)
        output = self.pool2(output)
        output = self.pad5(output)
        output = self.conv6(output)
        output = self.relu5(output)
        return output
    

class decoder3(nn.Module):
    def __init__(self):
        super(decoder3, self).__init__()
        
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)

        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
            
    def forward(self, input):
        output = self.pad1(input)
        output = self.conv7(output)
        output = self.relu1(output)
        output = self.unpool1(output)
        output = self.pad2(output)
        output = self.conv8(output)
        output = self.relu2(output)
        output = self.pad3(output)
        output = self.conv9(output)
        output = self.relu3(output)
        output = self.unpool2(output)
        output = self.pad4(output)
        output = self.conv10(output)
        output = self.relu4(output)
        output = self.pad5(output)
        output = self.conv11(output)
        return output
        

class encoder4(nn.Module):
    def __init__(self):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)
        
        self.pool2 = nn.MaxPool2d(2, 2)
            
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace = True)

        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace = True)

        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace = True)

        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace = True)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace = True)
            
    def forward(self, input):
        output = {} #存储对应层的输出以计算loss
        out = self.conv1(input)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.relu1(out)
        output['relu1_1'] = out
        out = self.pad2(out)
        out = self.conv3(out)
        out = self.relu2(out)
        output['relu1_2'] = out
        out = self.pool1(out)
        output['p1'] = out
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.relu3(out)
        output['relu2_1'] = out
        out = self.pad4(out)
        out = self.conv5(out)
        out = self.relu4(out)
        output['relu2_2'] = out
        out = self.pool2(out)
        output['p2'] = out
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.relu5(out)
        output['relu3_1'] = out
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.relu6(out)
        output['relu3_2'] = out
        out = self.pad7(out)
        out = self.conv8(out)
        out = self.relu7(out)
        output['relu3_3'] = out
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.relu8(out)
        output['relu3_4'] = out
        out = self.pool3(out)
        output['p3'] = out
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.relu9(out)
        output['relu4_1'] = out
        return output
    

class decoder4(nn.Module):
    def __init__(self):
        super(decoder4, self).__init__()

        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)
        
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace = True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace = True)

        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace = True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace = True)

        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)
            
    def forward(self, input):
        output = self.pad1(input)
        output = self.conv11(output)
        output = self.relu1(output)
        output = self.unpool1(output)
        output = self.pad2(output)
        output = self.conv12(output)
        output = self.relu2(output)
        output = self.pad3(output)
        output = self.conv13(output)
        output = self.relu3(output)
        output = self.pad4(output)
        output = self.conv14(output)
        output = self.relu4(output)
        output = self.pad5(output)
        output = self.conv15(output)
        output = self.relu5(output)
        output = self.unpool2(output)
        output = self.pad6(output)
        output = self.conv16(output)
        output = self.relu6(output)
        output = self.pad7(output)
        output = self.conv17(output)
        output = self.relu7(output)
        output = self.unpool3(output)
        output = self.pad8(output)
        output = self.conv18(output)
        output = self.relu8(output)
        output = self.pad9(output)
        output = self.conv19(output)
        return output
    

class encoder5(nn.Module):
    def __init__(self):
        super(encoder5, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)
        
        self.pool2 = nn.MaxPool2d(2, 2)
            
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace = True)

        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace = True)

        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace = True)

        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace = True)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace = True)

        self.pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace = True)

        self.pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace = True)

        self.pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace = True)

        self.pool4 = nn.MaxPool2d(2, 2)

        self.pad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace = True)
            
    def forward(self, input):
        output = {} #存储对应层的输出以计算loss
        out = self.conv1(input)
        out = self.pad1(out)
        out = self.conv2(out)
        out = self.relu1(out)
        output['relu1_1'] = out
        out = self.pad2(out)
        out = self.conv3(out)
        out = self.relu2(out)
        output['relu1_2'] = out
        out = self.pool1(out)
        output['p1'] = out
        out = self.pad3(out)
        out = self.conv4(out)
        out = self.relu3(out)
        output['relu2_1'] = out
        out = self.pad4(out)
        out = self.conv5(out)
        out = self.relu4(out)
        output['relu2_2'] = out
        out = self.pool2(out)
        output['p2'] = out
        out = self.pad5(out)
        out = self.conv6(out)
        out = self.relu5(out)
        output['relu3_1'] = out
        out = self.pad6(out)
        out = self.conv7(out)
        out = self.relu6(out)
        output['relu3_2'] = out
        out = self.pad7(out)
        out = self.conv8(out)
        out = self.relu7(out)
        output['relu3_3'] = out
        out = self.pad8(out)
        out = self.conv9(out)
        out = self.relu8(out)
        output['relu3_4'] = out
        out = self.pool3(out)
        output['p3'] = out
        out = self.pad9(out)
        out = self.conv10(out)
        out = self.relu9(out)
        output['relu4_1'] = out
        out = self.pad10(out)
        out = self.conv11(out)
        out = self.relu10(out)
        output['relu4_2'] = out
        out = self.pad11(out)
        out = self.conv12(out)
        out = self.relu11(out)
        output['relu4_3'] = out
        out = self.pad12(out)
        out = self.conv13(out)
        out = self.relu12(out)
        output['relu4_4'] = out
        out = self.pool4(out)
        output['p4'] = out
        out = self.pad13(out)
        out = self.conv14(out)
        out = self.relu13(out)
        output['relu5_1'] = out
        return output
    

class decoder5(nn.Module):
    def __init__(self):
        super(decoder5, self).__init__()

        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu1 = nn.ReLU(inplace = True)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace = True)

        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace = True)

        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace = True)

        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace = True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace = True)

        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace = True)

        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace = True)
        
        self.pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace = True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace = True)

        self.pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace = True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor = 2)

        self.pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace = True)

        self.pad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
            
    def forward(self, input):
        output = self.pad1(input)
        output = self.conv15(output)
        output = self.relu1(output)
        output = self.unpool1(output)
        output = self.pad2(output)
        output = self.conv16(output)
        output = self.relu2(output)
        output = self.pad3(output)
        output = self.conv17(output)
        output = self.relu3(output)
        output = self.pad4(output)
        output = self.conv18(output)
        output = self.relu4(output)
        output = self.pad5(output)
        output = self.conv19(output)
        output = self.relu5(output)
        output = self.unpool2(output)
        output = self.pad6(output)
        output = self.conv20(output)
        output = self.relu6(output)
        output = self.pad7(output)
        output = self.conv21(output)
        output = self.relu7(output)
        output = self.pad8(output)
        output = self.conv22(output)
        output = self.relu8(output)
        output = self.pad9(output)
        output = self.conv23(output)
        output = self.relu9(output)
        output = self.unpool3(output)
        output = self.pad10(output)
        output = self.conv24(output)
        output = self.relu10(output)
        output = self.pad11(output)
        output = self.conv25(output)
        output = self.relu11(output)
        output = self.unpool4(output)
        output = self.pad12(output)
        output = self.conv26(output)
        output = self.relu12(output)
        output = self.pad13(output)
        output = self.conv27(output)
        return output