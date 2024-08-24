import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# basic conv block
def conv_block(input_channels, n_output, stride=1, kernel_size=80):
    layers = []
    if stride == 1:
        layers.append(
            nn.Conv1d(
                input_channels,
                n_output,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
            )
        )  # Conv
    else:
        layers.append(
            nn.Conv1d(input_channels, n_output, kernel_size=kernel_size, stride=stride)
        )  # Conv
    layers.append(nn.BatchNorm1d(n_output))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# basic 2-conv residual block
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super().__init__()

        self.conv_block1 = conv_block(
            n_channels, n_channels, stride=1, kernel_size=kernel_size
        )
        self.conv_block2 = conv_block(n_channels, n_channels, stride=1, kernel_size=3)

    def forward(self, x):

        identity = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + identity
        return x


class ResNetRagaClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_channels = params.input_channels
        n_channel = params.n_channel
        stride = params.stride
        self.n_blocks = params.n_blocks
        self.conv_first = conv_block(
            input_channels, n_channel, stride=stride, kernel_size=80
        )
        self.max_pool_every = params.max_pool_every

        self.res_blocks = nn.ModuleList()  # Residual Blocks
        for i in range(self.n_blocks):
            self.res_blocks.append(ResidualBlock(n_channel, kernel_size=3))

        # linear classification head
        self.fc1 = nn.Linear(n_channel, params.num_classes)

    def forward(self, x):
        # initial conv
        x = self.conv_first(x)

        # residual blocks
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i % self.max_pool_every == 0:
                x = F.max_pool1d(x, 2)

        # classification head
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=-1)

        return x


class BaseRagaClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_channels = params.input_channels
        n_channel = params.n_channel
        stride = params.stride
        self.conv_blocks = []

        self.conv_block1 = conv_block(
            input_channels, n_channel, stride=stride, kernel_size=80
        )
        self.conv_block2 = conv_block(n_channel, n_channel, stride=1, kernel_size=3)
        self.conv_block3 = conv_block(n_channel, 2 * n_channel, stride=1, kernel_size=3)
        self.conv_block4 = conv_block(
            2 * n_channel, 2 * n_channel, stride=1, kernel_size=3
        )
        self.fc1 = nn.Linear(2 * n_channel, params.num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = F.max_pool1d(x, 4)
        x = self.conv_block2(x)
        x = F.max_pool1d(x, 4)
        x = self.conv_block3(x)
        x = F.max_pool1d(x, 4)
        x = self.conv_block4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Wav2VecTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.extractor_mode = params.extractor_mode
        self.extractor_conv_layer_config = params.extractor_conv_layer_config
        self.extractor_conv_bias = params.extractor_conv_bias
        self.encoder_embed_dim = params.encoder_embed_dim
        self.encoder_projection_dropout = params.encoder_projection_dropout
        self.encoder_pos_conv_kernel = params.encoder_pos_conv_kernel
        self.encoder_pos_conv_groups = params.encoder_pos_conv_groups
        self.encoder_num_layers = params.encoder_num_layers
        self.encoder_num_heads = params.encoder_num_heads
        self.encoder_attention_dropout = params.encoder_attention_dropout
        self.encoder_ff_interm_features = params.encoder_ff_interm_features
        self.encoder_ff_interm_dropout = params.encoder_ff_interm_dropout
        self.encoder_dropout = params.encoder_dropout
        self.encoder_layer_norm_first = params.encoder_layer_norm_first
        self.encoder_layer_drop = params.encoder_layer_drop
        self.aux_num_out = params.num_classes

        self.extractor_conv_layer_config = [
            (32, 80, 16),
            (64, 5, 4),
            (128, 5, 4),
            (256, 5, 4),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
        self.encoder = wav2vec2_model(
            self.extractor_mode,
            self.extractor_conv_layer_config,
            self.extractor_conv_bias,
            self.encoder_embed_dim,
            self.encoder_projection_dropout,
            self.encoder_pos_conv_kernel,
            self.encoder_pos_conv_groups,
            self.encoder_num_layers,
            self.encoder_num_heads,
            self.encoder_attention_dropout,
            self.encoder_ff_interm_features,
            self.encoder_ff_interm_dropout,
            self.encoder_dropout,
            self.encoder_layer_norm_first,
            self.encoder_layer_drop,
            aux_num_out=None,
        )

        self.audio_length = params.sample_rate * params.clip_length
        self.classification_head = nn.Linear(
            int(self.audio_length / (16 * 4 * 4 * 4 * 2 * 2 * 2))
            * params.encoder_embed_dim,
            params.num_classes,
        )

    def forward(self, x):
        x = self.encoder(x)[0]
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.classification_head(x)
        x = F.log_softmax(x, dim=-1)
        return x
