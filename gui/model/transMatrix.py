import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, layer, matrix_size = 32):
        super(CNN, self).__init__()

        #卷积层
        if layer == 'relu3_1':
            self.convs= nn.Sequential(
                nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features = input * expand_ratio, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features = input * expand_ratio, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(inplace = True), 
                nn.Conv2d(in_channels = 64, out_channels = matrix_size, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features= output, eps = 1e-05, momentum = 0.1, affine = True),
            )
        elif layer == 'relu4_1':
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features = input * expand_ratio, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(inplace = True),
                nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features = input * expand_ratio, eps = 1e-05, momentum = 0.1, affine = True),
                nn.ReLU(inplace = True), 
                nn.Conv2d(in_channels = 128, out_channels = matrix_size, kernel_size = 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(num_features= output, eps = 1e-05, momentum = 0.1, affine = True),
            )

        #全连接
        self.fc = nn.Linear(matrix_size * matrix_size, matrix_size * matrix_size)

    def forward(self, input):
        output = self.convs(input)
        #矩阵转向量
        b, c, h, w = output.size()
        output = output.view(b, c, -1)
        #计算协方差矩阵
        output = torch.bmm(output, output.transpose(1, 2)).div(h * w)
        output = self.fc(output.view(output.size(0), -1))
        return output
    

class TransModule(nn.Module):
    def __init__(self, layer, matrix_size = 32):
        super(TransModule, self).__init__()
        self.matrix_size = matrix_size

        self.c_cnn = CNN(layer, matrix_size)
        self.s_cnn = CNN(layer, matrix_size)

        if layer == 'relu3_1':
            self.compress = nn.Conv2d(in_channels = 256, out_channels = matrix_size, kernel_size = 1, stride = 1, padding = 0)
            self.uncompress = nn.Conv2d(in_channels = matrix_size, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)
        elif layer == 'relu4_1':
            self.compress = nn.Conv2d(in_channels = 512, out_channels = matrix_size, kernel_size = 1, stride = 1, padding = 0)
            self.uncompress = nn.Conv2d(in_channels = matrix_size, out_channels = 512, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, f_content, f_style):
        #去除颜色均值
        cb, cc, ch, cw = f_content.size()
        f_content_vec = f_content.view(cb, cc, -1)
        content_mean = torch.mean(f_content_vec, dim = 2, keepdim = True)
        content_mean = content_mean.unsqueeze(3).expand_as(f_content)
        f_content_norm = f_content - content_mean

        sb, sc, sh, sw = f_style.size()
        f_style_vec = f_style.view(sb, sc, -1)
        style_mean = torch.mean(f_style_vec, dim = 2, keepdim = True)
        style_mean_s = style_mean.unsqueeze(3).expand_as(f_style)
        style_mean_c = style_mean.unsqueeze(3).expand_as(f_content)
        f_style_norm = f_style - style_mean_s

        #transformation
        m_content = self.c_cnn(f_content_norm)
        m_style = self.s_cnn(f_style_norm)
        m_content = m_content.view(m_content.size(0), self.matrix_size, self.matrix_size)
        m_style = m_style.view(m_style.size(0), self.matrix_size, self.matrix_size)

        T = torch.bmm(m_style, m_content)

        #T与f_content_norm相乘
        f_c_compress = self.compress(f_content_norm)
        b, c, h, w = f_c_compress.size()
        f_c_compress = f_c_compress.view(b, c, -1)

        f_d_compress = torch.bmm(T, f_c_compress).view(b, c, h, w)
        f_d_norm = self.uncompress(f_d_compress)
        f_d = f_d_norm + style_mean_c

        return f_d, T
        