import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    '''
    ResNet block with LayerNorm
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, img_size=16):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param img_size: image size
        '''
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm([out_channels, img_size, img_size])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm([out_channels, img_size, img_size])

        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.adjust_channels:
            identity = self.adjust_channels(identity)
        return out + identity


class FeatureExtractor(nn.Module):
    '''
    Convolutional feature extractor with ResNet blocks
    '''

    def __init__(self, in_channels, num_features, depth, img_size=16):
        '''
        :param in_channels: number of input channels
        :param num_features: number of output channels
        :param depth: number of ResNet blocks
        :param img_size: image size
        '''
        super(FeatureExtractor, self).__init__()
        layers = [ResNetBlock(in_channels, num_features, img_size=img_size)]
        for _ in range(depth - 1):
            layers.append(ResNetBlock(num_features, num_features, img_size=img_size))
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    '''
    :param model: model
    return: number of trainable parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerEncoderLayer(nn.Module):
    '''
    Transformer encoder layer
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        '''
        :param d_model: number of features in the input
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward network
        :param dropout: dropout rate
        '''
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.norm1(attn_output)
        linear_output = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.norm2(linear_output)
        return x


class TransformerEncoder(nn.Module):
    '''
    Transformer encoder
    '''

    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        '''
        :param num_layers: number of layers in the encoder
        :param d_model: number of features in the input
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward network
        :param dropout: dropout rate
        '''
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EnhancedTemporalAttention(nn.Module):
    '''
    Temporal Attention
    '''

    def __init__(self, num_days, num_features, num_layers=2, nhead=2, dim_feedforward=2048, dropout=0.1):
        '''
        :param num_days: number of days or time steps
        :param num_features: number of features
        :param num_layers: number of layers in the encoder
        :param nhead: number of attention heads
        :param dim_feedforward: dimension of the feedforward network
        :param dropout: dropout rate
        '''
        super(EnhancedTemporalAttention, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_days, num_features))
        self.transformer_encoder = TransformerEncoder(num_layers, num_features, nhead, dim_feedforward, dropout)

    def forward(self, x):
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        return x


class DRQModel(nn.Module):
    '''
    Remote sensing model for discharge prediction
    '''

    def __init__(self, in_channels=2, num_days=3, img_size=16, num_features=64, depth=3):
        '''
        :param in_channels: number of input channels 
        :param num_days: number of days or time steps 
        :param img_size: image size
        :param num_features: number of features in the feature extractor
        :param depth: number of ResNet blocks in the feature extractor
        '''
        super(DRQModel, self).__init__()

        self.num_features = num_features
        self.num_days = num_days
        self.extractor = FeatureExtractor(in_channels, num_features, depth, img_size)
        self.attention = EnhancedTemporalAttention(num_days, num_features, num_layers=10, nhead=8, dim_feedforward=512,
                                                   dropout=0.1)
        self.fc = nn.Linear(num_features * num_days, num_days)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        features = self.extractor(x)
        features_combined = features.view(batch_size, self.num_days, self.num_features)
        attn_output = self.attention(features_combined)
        out = self.fc(attn_output.view(batch_size, self.num_days * self.num_features))
        return out
