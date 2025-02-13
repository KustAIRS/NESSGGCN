import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SSConv(nn.Module):   

    def __init__(self, in_ch, out_ch, kernel_size=3):    # (batch_size, in_ch, height, width)
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(  # (batch_size, out_ch, height, width)
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(    # (batch_size, out_ch, height, width)  1*1卷积核
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)


    def forward(self, input):
        out = self.point_conv(self.BN(input))
        # pdb.set_trace()
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

def CalSupport(A):
    num1 = A.shape[0]
    A_ = A + 15 * torch.eye(num1, device=A.device)  # Add self-loops
    D_ = torch.sum(A_, dim=1)
    D_05 = torch.diag(torch.pow(D_, -0.5))
    support = torch.matmul(torch.matmul(D_05, A_), D_05)
    return support




class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, H, support):
        out = torch.matmul(support, H)
        out = self.linear(out)
        return F.relu(out)

class GatedGCNBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=8/3, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        
        self.gcn_layer = GCNLayer(conv_channels, conv_channels)

        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, support):
        shortcut = x  # [B, N, F]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        
        c = self.gcn_layer(c, support)
        
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut



class NESSGGCN(nn.Module):
    def __init__(self, height, width, channels, class_count, Q, A_euclidean, A_binary):
        super(GGCN, self).__init__()
        self.class_count = class_count  # 类别数
        self.channels = channels
        self.height = height
        self.width = width
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q
        self.A_euclidean = CalSupport(A_euclidean)  # 归一化邻接矩阵
        self.A_binary = CalSupport(A_binary)

        layers_count = 2

        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channels))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(self.channels, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=3))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))


        self.gated_gcn1 = GatedGCNBlock(128)
        self.gated_gcn2 = GatedGCNBlock(128)

        self.linear1 = nn.Linear(128, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise

        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)
        hx = clean_x

        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])


        H = self.gated_gcn1(superpixels_flatten, self.A_euclidean)
        H = self.gated_gcn2(H, self.A_binary)

        GCN_result = torch.matmul(self.Q, H)  
        GCN_result = self.linear1(GCN_result)
        GCN_result = self.act1(self.bn1(GCN_result))
        Y = 0.4 * CNN_result + 0.6 * GCN_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y






