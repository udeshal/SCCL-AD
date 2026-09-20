import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TabTransformNet(nn.Module):
    def __init__(self, x_dim,h_dim,num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            #print('Inputdim1: ', input_dim)
            #print('hdim1: ', h_dim)
            net.append(nn.Linear(input_dim,h_dim,bias=False))
            # net.append(nn.BatchNorm1d(h_dim,affine=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        #print('Inputdim2: ', input_dim)
        #print('xdim1: ', x_dim)
        net.append(nn.Linear(input_dim,x_dim,bias=False))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        #print('Trans X : ',x.shape )
        out = self.net(x)

        return out


class TabEncoder(nn.Module):
    def __init__(self, x_dim,h_dim,z_dim,bias,num_layers,batch_norm):

        super(TabEncoder, self).__init__()

        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim,bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim,affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim,bias=bias)
    def forward(self, x):

        z = self.enc(x)
        z = self.fc(z)

        return z
class TabScaleNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, reduce=16):
        super(TabScaleNet, self).__init__()
        for _ in range(num_layers - 1):
          #kernel_size 3,5,7 and padding 1,2,3
          self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
          self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=6, padding='same')
          self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=12, padding='same')
          
          self.bn = nn.BatchNorm1d(out_channels * 3)
          self.reduce = reduce
          self.fc1 = nn.Linear(out_channels * 3, out_channels * 3 // reduce, bias=False)
          self.fc2 = nn.Linear(out_channels * 3 // reduce, out_channels * 3, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        print('X Shape in Scale : ', x.shape)
        if x.dim() == 2:
          x = x.unsqueeze(0)
        
        print('X Shape in Scale2 : ', x.shape)

        x = x.permute(0, 2, 1)  # -> (batch, channels, seq_len)
        print('X Shape in Scale3 : ', x.shape)

        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        #for arrhythmia when kernel size is changed
        #out2 =  out2[:, :, :128]  # Keeps first 128 elements along last dimension
        #F.pad(out2, (0, 2))
        #out3 = out3[:, :, :128] #F.pad(out3, (0, 8))

        print('Out Shape in Scale1 : ', out1.shape)
        print('out Shape in Scale2 : ', out2.shape)
        print('out Shape in Scale3 : ', out3.shape)

        out = torch.cat([out1, out2, out3], dim=1)  # concat along channel dim
        out = self.bn(out)
        out = F.relu(out)

        # Squeeze-and-Excitation
        y = torch.mean(out, dim=2)  # (batch, channels)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).unsqueeze(2)  # (batch, channels, 1)

        out = out * y  # channel-wise scaling
        out = out.permute(0, 2, 1)  # -> (batch, seq_len, channels)        
        out.squeeze(0);
        #this code required only if the dataset is arrythmia
        out = F.pad(out, (0, 2)) #to make the channel size to 274 for arrythmia (91*3=273 only), 38 for SMD (12*3 = 36 only)
        return out

class ScaleAttentionNet(nn.Module):
    #kernels=91 for Arrythmia and 2 for Thyroid and 12 for SMD
    def __init__(self, input_dim, num_classes, num_layers, num_blocks=1, kernels=12, reduce=16):
        super(ScaleAttentionNet, self).__init__()
        self.blocks = nn.Sequential(*[
            TabScaleNet(input_dim if i == 0 else kernels * 3, kernels, reduce, num_layers)
            for i in range(num_blocks)
        ])
        #self.classifier = nn.Linear(kernels * 3, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        #x = torch.mean(x, dim=1)  # global average pooling over sequence
        #logits = self.classifier(x)
        return x
class MACNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=16, kernel_sizes=[3, 6, 12]):
        super(MACNNBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.fc1 = nn.Linear(out_channels * len(kernel_sizes), out_channels * len(kernel_sizes) // reduce, bias=False)
        self.fc2 = nn.Linear(out_channels * len(kernel_sizes) // reduce, out_channels * len(kernel_sizes), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, T, C] → convert to [B, C, T] for Conv1D
        if x.dim() == 2:
          x = x.unsqueeze(0)
        #print('X1 : ', x.shape)
        x = x.transpose(1, 2)  # [B, C_in, T]
        #print('X2 : ', x.shape)

        convs = [conv(x) for conv in self.convs]  # each output: [B, C_out, T]
        #print('conv1 : ', convs[0].shape)
        #print('conv2 : ', convs[1].shape)
        #print('conv3 : ', convs[2].shape)
        x = torch.cat(convs, dim=1)              # [B, C_out * len(kernel_sizes), T]
        x = self.bn(x)
        x = F.relu(x)

        # Channel attention
        y = torch.mean(x, dim=2)  # Global Average Pooling: [B, C]
        y = F.relu(self.fc1(y))   # [B, C//r]
        y = self.sigmoid(self.fc2(y))  # [B, C]
        y = y.unsqueeze(2)  # [B, C, 1]

        x = x * y  # Attention applied
        x = x.transpose(1, 2)  # Back to [B, T, C]
        return x


class MACNNNet(nn.Module):
    def __init__(self, input_dim, classes_num):
        super(MACNNNet, self).__init__()
        #for psm self.stack1 = self._make_stack(input_dim, 8, loop_num=2)
        #for SMD - 
        self.stack1 = self._make_stack(input_dim, 12, loop_num=2)
        #for Arrhythmia - self.stack1 = self._make_stack(input_dim, 91, loop_num=2)
        # for thyroid - self.stack1 = self._make_stack(input_dim, 2, loop_num=2)
        # for SMAP -  self.stack1 = self._make_stack(input_dim, 400, loop_num=2)
        #for MSL  -   self.stack1 = self._make_stack(input_dim, 421, loop_num=2)
        #for SMAP full data load (25 channels) self.stack1 = self._make_stack(input_dim, 8, loop_num=2)
        #for swat self.stack1 = self._make_stack(input_dim, 17, loop_num=2)
        #for wadi self.stack1 = self._make_stack(input_dim, 42, loop_num=2)

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        #self.stack2 = self._make_stack(64 * 3, 128, loop_num=2)
        #self.stack2 = self._make_stack(12 * 3, 128, loop_num=2)
        #self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # replaces tf.reduce_mean(x, 1)
        #self.classifier = nn.Linear(128 * 3, classes_num)

    def _make_stack(self, in_channels, out_channels, loop_num=2):
        layers = []
        for _ in range(loop_num):
            #print('In channels : ', in_channels)
            #print('Out channels : ', out_channels)
            layers.append(MACNNBlock(in_channels, out_channels))
            in_channels = out_channels * 3  # updated after first MACNNBlock
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, T, input_dim]
        #print('Mac1 X1: ', x.shape)
        x = self.stack1(x)
        x = x.transpose(1, 2)  # [B, C, T] for pooling
        #print('Mac1 X2: ', x.shape)
        x = self.pool1(x)
        x = x.transpose(1, 2)  # back to [B, T, C]

        #for SMD 
        x = F.pad(x, (0, 2)) #make 38 (12*3 +2)
        #for MSL  x = F.pad(x, (0, 2)) #make 1265 (421*3 + 2)
        #for SMAP x = F.pad(x, (0,0)) #for full data load

        #for PSM  x = F.pad(x, (0, 1))
        #for swat x = F.pad(x, (0,0))
        #for wadi x = F.pad(x, (0, 1))

        #print('Mac1 X3: ', x.shape)

        #x = self.stack2(x)
        #print('Mac2 X1: ', x.shape)
        #x = x.transpose(1, 2)  # [B, C, T]
        #x = self.global_avg_pool(x).squeeze(2)  # [B, C]
        #print('Mac2 X2: ', x.shape)

        #logits = self.classifier(x)  # [B, classes_num]
        #print('logits : ', logits.shape)
        return x

class TabNets():

    def _make_nets(self,x_dim,config):
        enc_nlayers = config['enc_nlayers']
        try:
            hdim = config['enc_hdim']
            zdim = config['latent_dim']
            trans_hdim = config['trans_hdim']
        except:
            if 32<=x_dim <= 300:
                zdim = 32
                hdim = 64
                trans_hdim = x_dim
            elif x_dim<32:
                zdim = 2 * x_dim
                hdim = 2 * x_dim
                trans_hdim = x_dim
            else:
                zdim = 64
                hdim = 256
                trans_hdim = x_dim
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']
        #scales = nn.ModuleList([ScaleAttentionNet(x_dim,2, enc_nlayers) for _ in range(num_trans)])
        
        #commented for ablation scales = nn.ModuleList([MACNNNet(x_dim, 2) for _ in range(num_trans)])

        enc = TabEncoder(x_dim, hdim,zdim, config['enc_bias'],enc_nlayers,batch_norm)
        trans = nn.ModuleList(
            [TabTransformNet(x_dim, trans_hdim, trans_nlayers) for _ in range(num_trans)])


        return enc,trans #commented for ablation ,scales
