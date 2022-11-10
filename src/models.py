"""
    Models in use for this project.
"""

import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BottleNeck(nn.Module):
    """
        ConvNext Reversed BottleNeck Block
    """
    def __init__(
            self, 
            dim: int=64,
            useLayerNorm: bool=False
        ):
        super().__init__()
        # automatically set the output dims to be C, 4 * C, C 
        self.dim = dim
        # depthwise 7x7 conv
        self.dwconv = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=7,
            padding=3,
            groups=self.dim
        )
        self.norm = (
            nn.LayerNorm(self.dim) if useLayerNorm 
            else nn.BatchNorm1d(self.dim)
        )
        self.pwconv1 = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim * 4,
            kernel_size=1
        )
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv1d(
            in_channels=self.dim * 4,
            out_channels=self.dim,
            kernel_size=1
        )
    

    def forward(self, x):
        # keep copy
        identity = x

        # feed forward
        xx = self.dwconv(x)
        xx = self.norm(xx)
        xx = self.pwconv1(xx)
        xx = self.activation(xx)
        xx = self.pwconv2(xx)

        # residual connection
        xx += identity

        return xx



class FeatureExtraction(nn.Module):
    """
        A CNN-Empowered Feature Extraction Head for MFCC Features
    """
    def __init__(
            self,
            dim_in: int=15,
            dims_out: list=[128, 256, 512],
            strides: list=[2, 2, 1, 1], 
            kernels: list=[5, 5, 5, 3],
            useLayerNorm: bool=True
        ):
        super().__init__()

        self.dim_in = dim_in
        self.dims_out = dims_out
        self.strides = strides
        self.kernels = kernels
        self.paddings = [k // 2 for k in self.kernels]
        assert (
            len(self.dims_out) + 1 
            == len(self.strides) 
            == len(self.kernels)
        )
        
        self.shrinks = nn.ModuleList()
        # first shrinking conv1d layer
        # DIM: (batch_size, sequence_length, dim_in)
        init_shrink = nn.Sequential(
            nn.Conv1d(
                in_channels=self.dim_in,
                out_channels=self.dims_out[0],
                kernel_size=self.kernels[0],
                stride=self.strides[0],
                padding=self.paddings[0]
            ),
            nn.BatchNorm1d(self.dims_out[0]),
            nn.GELU()
        )
        # DIM: (batch_size, sequence_length*, dims_out[0])
        self.shrinks.append(init_shrink)

        # ConvNext Blocks & Length Shrinking Blocks
        self.convs = nn.ModuleList()
        for i, out in enumerate(self.dims_out):
            conv = BottleNeck(
                dim=out,
                useLayerNorm=useLayerNorm
            )
            self.convs.append(conv)
            if len(self.shrinks) < len(self.dims_out):
                shrink = nn.Sequential(
                    nn.Conv1d(
                        in_channels=out,
                        out_channels=self.dims_out[i + 1],
                        stride=self.strides[i + 1],
                        kernel_size=self.kernels[i + 1]
                    ),
                    nn.BatchNorm1d(self.dims_out[i + 1]),
                    nn.GELU()
                )
                self.shrinks.append(shrink)
        # DIM: (batch_size, sequence_length**, dims_out[-1])
        
        # calculate how much discount applied to the lengths
        self.len_discount = 1
        for s in self.strides:
            self.len_discount *= s

    
    def forward(self, x, lx):
        xx = x.permute((0, 2, 1))
        # iterative operation
        for i, shrink in enumerate(self.shrinks):
            conv = self.convs[i]
            xx = shrink(xx)
            xx = conv(xx)
        xx = xx.permute((0, 2, 1))
        # compute new lengths
        lx = torch.clamp(lx // self.len_discount, max=xx.shape[1])
        return xx, lx



class AccrualNet(nn.Module):
    """
        (Bidirectional) LSTM Module
    """
    def __init__(
            self,
            dim_in: int=512,
            hidden_dims: list=[256, 256],
            num_layers: list=[4, 4],
            bidirectionals: list=[True, True],
            dropouts: list=[0.3, 0.3]
        ):
        super().__init__()

        self.dims = [dim_in] + hidden_dims
        self.num_layers = num_layers
        self.bidirectionals = bidirectionals
        assert (
            len(self.dims) 
            == len(self.num_layers) + 1 
            == len(self.bidirectionals) + 1
        )

        self.streamline = nn.ModuleList()
        for l, dim in enumerate(self.dims[:-1]):
            bidirectional = self.bidirectionals[l]
            input_dim = (
                dim * 2 if l > 0 and self.bidirectionals[l]
                else dim
            )
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.dims[l + 1],
                num_layers=self.num_layers[l],
                bidirectional=bidirectional,
                dropout=dropouts[l],
                batch_first=True
            )
            self.streamline.append(lstm)
    

    def forward(self, x, lx):
        xx = pack_padded_sequence(
            x, lengths=lx, batch_first=True, enforce_sorted=False
        )
        for layer in self.streamline:
            xx, _ = layer(xx)
        xx, lx = pad_packed_sequence(
            xx, batch_first=True
        )
        return xx, lx



class ClsHead(nn.Module):
    """
        Final Linear & Classification Module
    """
    def __init__(
            self, 
            dim_in: int=256,
            dims: list=[512],
            num_labels: int=41
        ):
        super().__init__()
        # bookkeeping
        self.dims = [dim_in] + dims + [num_labels]
        self.linears = nn.ModuleList()
        # linear(s)
        for i, dim in enumerate(self.dims[:-1]):
            self.linears.append(
                nn.Linear(
                    in_features=dim,
                    out_features=self.dims[i + 1]
                )
            )
            if i < len(self.dims) - 2:
                self.linears.append(nn.GELU())
        assert len(self.linears) == 2 * len(self.dims) - 3
        # logsoftmax
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        x = self.logsoftmax(x)
        return x



class OneForAll(nn.Module):
    """
        Entire network from feature extraction to final classification.
    """
    def __init__(
            self, 
            feat_ext_cfgs: dict,
            lstm_cfgs: dict,
            cls_cfgs: dict
        ):
        super().__init__()
        self.configs = {
            'feat_ext': feat_ext_cfgs,
            'lstm': lstm_cfgs,
            'cls': cls_cfgs
        }
        self.feat_ext = FeatureExtraction(**feat_ext_cfgs)
        # dim_in for lstm is the final output channel of CNNs
        lstm_dim_in = feat_ext_cfgs['dims_out'][-1]
        self.lstm = AccrualNet(dim_in=lstm_dim_in, **lstm_cfgs)
        # dim_in for cls is the final hidden dimension of LSTMs
        cls_dim_in = lstm_cfgs['hidden_dims'][-1]
        # alter the dimension based on whether bidirectional is enabled
        cls_dim_in *= 2 if lstm_cfgs['bidirectionals'][-1] else 1
        self.cls = ClsHead(dim_in=cls_dim_in, **cls_cfgs)
    

    def forward(self, x, lx):
        # feature extraction layers
        xx, lx = self.feat_ext(x, lx)

        # (bi)lstm layers
        xx, lx = self.lstm(xx, lx)

        # cls layers
        xx = self.cls(xx)

        return xx, lx



"""
    Main function below is for local testing purpose only.
""" 
if __name__ == '__main__':
    
    B = 4
    MINLEN, MAXLEN = 800, 2000
    MFCC_DIM = 15
    DIMS_OUT = [16, 32, 32]
    KERNELS = [5, 5, 5, 3]
    STRIDES = [2, 2, 1, 1]
    LSTM_DIMOUT = 64
    NUM_LAYERS = [2, 2]
    BIDIRECTIONALS = [True, True]
    DROPOUTS = [0.3, 0.3]
    LINEAR_DIMS = [64]
    NUM_PHONEMES = 41

    LX = torch.tensor([
        torch.randint(MINLEN, MAXLEN, size=()) 
        for _ in range(B)
    ])
    X = [
        torch.rand(size=(LX[i], MFCC_DIM))
        for i in range(B)
    ]
    # pad
    X = nn.utils.rnn.pad_sequence(X).permute((1, 0, 2))
    # (batch size, channels, sequence length)
    print('Input: ', X.shape)

    FeatExt = FeatureExtraction(
        dim_in=MFCC_DIM,
        dims_out=DIMS_OUT,
        strides=STRIDES,
        kernels=KERNELS,
        useLayerNorm=False
    )
    print(f"\nModel summary for [Feature Extraction]: \n{summary(FeatExt, X, LX)}\n\n")

    XX, LX = FeatExt(X, LX)
    print('Post-Feature-Extraction: ', XX.shape)

    BiLSTM = AccrualNet(
        dim_in=DIMS_OUT[-1],
        hidden_dims=HIDDEN_DIMS,
        num_layers=NUM_LAYERS,
        bidirectionals=BIDIRECTIONALS,
        dropouts=DROPOUTS
    )
    print(f"\nModel summary for [BiLSTM]: \n{summary(BiLSTM, XX, LX)}\n\n")

    XX, LX = BiLSTM(XX, LX)
    print('Post-LSTM: ', XX.shape)


    ClsBlock = ClsHead(
        dim_in=HIDDEN_DIMS[-1] * (1 + int(BIDIRECTIONALS[-1])),
        dims=LINEAR_DIMS,
        num_labels=NUM_PHONEMES
    )
    print(f"\nModel summary for [CLS]: \n{summary(ClsBlock, XX)}\n\n")

    XX = ClsBlock(XX)
    print('Post-Classification: ', XX.shape)


    """
        Compare w/ building the entire network end-to-end.
    """
    MFCC_DIM = 15
    DIMS_OUT = [16, 32, 32]
    KERNELS = [5, 5, 5, 3]
    STRIDES = [2, 2, 1, 1]
    HIDDEN_DIMS = [32, 32]
    NUM_LAYERS = [2, 2]
    BIDIRECTIONALS = [True, True]
    LINEAR_DIMS = [64]
    NUM_PHONEMES = 41

    FEAT_EXT_CFGS = {
        'dim_in': MFCC_DIM,
        'dims_out': DIMS_OUT,
        'strides': STRIDES,
        'kernels': KERNELS,
        'useLayerNorm': False
    }

    LSTM_CFGS = {
        'hidden_dims': HIDDEN_DIMS,
        'num_layers': NUM_LAYERS,
        'bidirectionals': BIDIRECTIONALS,
        'dropouts': DROPOUTS
    }

    CLS_CFGS = {
        'dims': LINEAR_DIMS,
        'num_labels': NUM_PHONEMES
    }

    model = OneForAll(
        feat_ext_cfgs=FEAT_EXT_CFGS,
        lstm_cfgs=LSTM_CFGS,
        cls_cfgs=CLS_CFGS
    )

    print(f"\n\nModel Summary for [OneForAll]:\n{summary(model, X, LX)}\n\n")