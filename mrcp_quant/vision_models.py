import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .bert_model import DEFAULT_MODULE_QUANT_PARAMS
from .layers import (
    IntGeluTS,
    IntSoftmaxTS,
    QLayerNorm,
    QuantizedLinear,
    QuantizedMatmul,
    qHadamardProd,
)


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std)


def _apply_quant_defaults(module):
    for key, value in DEFAULT_MODULE_QUANT_PARAMS.items():
        setattr(module, key, value)


class QauntParams(nn.Module):
    def __init__(self):
        super().__init__()
        _apply_quant_defaults(self)


class QuantTransformer(QauntParams):
    def __init__(self, quant=False, is_calibrate=False, q_module_list=None):
        super().__init__()
        self.quant = quant
        self.is_calibrate = is_calibrate
        self.q_module_list = list(q_module_list or [])

    def set_q_module_list(self, q_module_list):
        self.q_module_list = list(q_module_list)

    def set_calibration_flag(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_calibration_flag()

    def unset_calibration_flag(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_calibration_flag()

    def set_quant(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_quant()

    def unset_quant(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_quant()

    def set_scale_opt(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.set_scale_opt()

    def unset_scale_opt(self):
        for module in self.modules():
            if type(module) in self.q_module_list:
                module.unset_scale_opt()


class Mlp(QauntParams):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=IntGeluTS,
        quant=False,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QuantizedLinear(
            in_features,
            hidden_features,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.act = act_layer(
            quant=quant,
            LUT_SIZE=self.lut_size_gelu,
            nof_bits=self.nof_bits_gelu,
            split_table=self.split_table_gelu,
        )
        self.fc2 = QuantizedLinear(
            hidden_features,
            out_features,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(QauntParams):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, quant=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = QuantizedLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.mat_mul_qk = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.sf = IntSoftmaxTS(
            nof_bits=self.nof_bits_softmax,
            LUT_SIZE=self.lut_size_softmax,
            split_table=self.split_table_softmax,
            dim=-1,
            quant=quant,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.mat_mul_pv = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.proj = QuantizedLinear(
            dim,
            dim,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = self.mat_mul_qk(query, key.transpose(-2, -1)) * self.scale
        attn = self.sf(attn)
        attn = self.attn_drop(attn)
        x = self.mat_mul_pv(attn, value).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(QauntParams):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=IntGeluTS,
        norm_layer=QLayerNorm,
        quant=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            quant=quant,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            quant=quant,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DeiTPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DistilledVisionTransformer(QuantTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=True,
        quant=False,
        is_calibrate=False,
        q_module_list=None,
        **kwargs,
    ):
        super().__init__(quant=quant, is_calibrate=is_calibrate, q_module_list=q_module_list)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.patch_embed = DeiTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=kwargs.get("drop_rate", 0.0))
        self.blocks = nn.ModuleList([
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=IntGeluTS,
                norm_layer=QLayerNorm,
                quant=quant,
            )
            for _ in range(depth)
        ])
        self.norm = QLayerNorm(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.head = QuantizedLinear(
            embed_dim,
            num_classes,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        ) if num_classes > 0 else nn.Identity()
        self.head_dist = QuantizedLinear(
            embed_dim,
            num_classes,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        ) if distilled else None
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, QLayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.dist_token is None:
            return x[:, 0]
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x_cls, x_dist = x
            x = self.head(x_cls), self.head_dist(x_dist)
            if not self.training:
                return (x[0] + x[1]) / 2
            return x
        return self.head(x)


def window_partition(x, window_size):
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, channels
    )


def window_reverse(windows, window_size, height, width):
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)


class WindowAttention(QauntParams):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        quant=False,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = QuantizedLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.mat_mul_qk = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.mat_mul_pv = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.proj = QuantizedLinear(
            dim,
            dim,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = IntSoftmaxTS(
            nof_bits=self.nof_bits_softmax,
            LUT_SIZE=self.lut_size_softmax,
            split_table=self.split_table_softmax,
            dim=-1,
            quant=quant,
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, num_tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = self.mat_mul_qk(query, key.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                batch_windows // num_windows, num_windows, self.num_heads, num_tokens, num_tokens
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = self.mat_mul_pv(attn, value).transpose(1, 2).reshape(
            batch_windows, num_tokens, channels
        )
        x = self.proj(x)
        return self.proj_drop(x)

    def flops(self, num_tokens):
        flops = num_tokens * self.dim * 3 * self.dim
        flops += self.num_heads * num_tokens * (self.dim // self.num_heads) * num_tokens
        flops += self.num_heads * num_tokens * num_tokens * (self.dim // self.num_heads)
        flops += num_tokens * self.dim * self.dim
        return flops


class SwinTransformerBlock(QauntParams):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=IntGeluTS,
        norm_layer=QLayerNorm,
        quant=False,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in 0-window_size")
        self.norm1 = norm_layer(
            dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            quant=quant,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(
            dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            quant=quant,
        )
        if self.shift_size > 0:
            height, width = self.input_resolution
            img_mask = torch.zeros((1, height, width, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = count
                    count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        height, width = self.input_resolution
        batch_size, length, channels = x.shape
        if length != height * width:
            raise ValueError("input feature has wrong size")
        shortcut = x
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(batch_size, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def flops(self):
        height, width = self.input_resolution
        flops = self.dim * height * width
        num_windows = height * width / self.window_size / self.window_size
        flops += num_windows * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * height * width * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * height * width
        return flops


class PatchMerging(QauntParams):
    def __init__(self, input_resolution, dim, quant=False, norm_layer=QLayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = QuantizedLinear(
            4 * dim,
            2 * dim,
            bias=False,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.norm = norm_layer(
            4 * dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )

    def forward(self, x):
        height, width = self.input_resolution
        batch_size, length, channels = x.shape
        if length != height * width:
            raise ValueError("input feature has wrong size")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"x size ({height}*{width}) must be even")
        x = x.view(batch_size, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(batch_size, -1, 4 * channels)
        return self.reduction(self.norm(x))

    def flops(self):
        height, width = self.input_resolution
        flops = height * width * self.dim
        flops += (height // 2) * (width // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(QauntParams):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=QLayerNorm,
        downsample=None,
        use_checkpoint=False,
        quant=False,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                act_layer=IntGeluTS,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (index % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                quant=quant,
                fused_window_process=fused_window_process,
            )
            for index in range(depth)
        ])
        self.downsample = (
            downsample(input_resolution, dim=dim, norm_layer=norm_layer, quant=quant)
            if downsample is not None
            else None
        )

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = sum(block.flops() for block in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinPatchEmbed(QauntParams):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, quant=False, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = (
            norm_layer(
                embed_dim,
                in1_bits=self.nof_bits_lnorm1,
                in2_bits=self.nof_bits_lnorm2,
                split_table=self.split_table_lnorm,
                quant=quant,
            )
            if norm_layer is not None
            else None
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        height_out, width_out = self.patches_resolution
        flops = height_out * width_out * self.embed_dim * self.in_chans
        flops *= self.patch_size[0] * self.patch_size[1]
        if self.norm is not None:
            flops += height_out * width_out * self.embed_dim
        return flops


class SwinTransformer(QuantTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=QLayerNorm,
        ape=False,
        patch_norm=True,
        is_calibrate=False,
        quant=False,
        use_checkpoint=False,
        fused_window_process=False,
        q_module_list=None,
        **kwargs,
    ):
        super().__init__(quant=quant, is_calibrate=is_calibrate, q_module_list=q_module_list)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = SwinPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            quant=quant,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for layer_index in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** layer_index),
                input_resolution=(
                    patches_resolution[0] // (2 ** layer_index),
                    patches_resolution[1] // (2 ** layer_index),
                ),
                depth=depths[layer_index],
                num_heads=num_heads[layer_index],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:layer_index]):sum(depths[:layer_index + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (layer_index < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                quant=quant,
                fused_window_process=fused_window_process,
            )
            self.layers.append(layer)
        self.norm = norm_layer(
            self.num_features,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            split_table=self.split_table_lnorm,
            quant=quant,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = QuantizedLinear(
            self.num_features,
            num_classes,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        ) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, QuantizedLinear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, QLayerNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def flops(self):
        flops = self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1]
        flops += self.num_features * self.num_classes
        return flops


def deit_base_patch16_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        quant=quant,
        q_module_list=q_module_list,
        distilled=False,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"])
    return model


def deit_small_patch16_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        q_module_list=q_module_list,
        quant=quant,
        distilled=False,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"])
    return model


def deit_tiny_patch16_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        q_module_list=q_module_list,
        quant=quant,
        distilled=False,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"])
    return model


def swin_tiny_patch4_window7_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        quant=quant,
        norm_layer=QLayerNorm,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"], strict=False)
    return model


def swin_small_patch4_window7_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        norm_layer=QLayerNorm,
        embed_dim=96,
        quant=quant,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"], strict=False)
    return model


def swin_base_patch4_window7_224(pretrained=False, quant=False, q_module_list=None, **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        norm_layer=QLayerNorm,
        quant=quant,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )
    if pretrained:
        checkpoint_data = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
            map_location="cpu",
        )
        model.load_state_dict(checkpoint_data["model"], strict=False)
    return model


MODEL_LOADERS = {
    "deit_tiny_patch16_224": deit_tiny_patch16_224,
    "deit_small_patch16_224": deit_small_patch16_224,
    "deit_base_patch16_224": deit_base_patch16_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
}


def get_vision_model_loader(model_name):
    if model_name not in MODEL_LOADERS:
        valid = ", ".join(sorted(MODEL_LOADERS))
        raise ValueError(f"Unknown vision model {model_name!r}. Valid models: {valid}")
    return MODEL_LOADERS[model_name]

