import torch
from torch import nn

import math
import warnings

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearProject(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super(LinearProject, self).__init__()
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()
        self.fn = fn

    # import torchsnooper
    # @torchsnooper.snoop()
    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads,  attn_drop=0., proj_drop=0.):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, complement):

        # x [B, HW, C]
        B_x, N_x, C_x = x.shape

        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, HW, C//num_heads]
        q = self.to_q(x).reshape(B_x, N_x, self.num_heads, C_x//self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, self.num_heads, C_c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 1., attn_drop=0., proj_drop=0.,drop_path = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer(dim)
        self.c_norm1 = norm_layer(dim)

        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop, proj_drop)

        self.x_norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(drop_path)
        self.drop2 = nn.Dropout(drop_path)

    def forward(self, x, complement):
        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))
        x = x + self.drop2(self.mlp(self.x_norm2(x)))
        return x


class CrossTransformer(nn.Module):
    def __init__(self, x_dim, c_dim, depth, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(CrossTransformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearProject(x_dim, c_dim, CrossTransformerEncoderLayer(c_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path)),
                LinearProject(c_dim, x_dim, CrossTransformerEncoderLayer(x_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path))
            ]))

    def forward(self, x, complement):
        x = x.unsqueeze(1)
        complement = complement.unsqueeze(1)
        for x_attn_complemnt, complement_attn_x in self.layers:
            x = x_attn_complemnt(x, complement=complement) + x
            complement = complement_attn_x(complement, complement=x) + complement
        return x.squeeze(1), complement.squeeze(1)

class MetaFusion(nn.Module):
    def __init__(self, derm_dim, cli_dim, meta_dim, depth, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(MetaFusion, self).__init__()
        self.proj_in = nn.Linear(meta_dim, derm_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearProject(derm_dim, derm_dim, CrossTransformerEncoderLayer(derm_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path)),
                LinearProject(cli_dim, cli_dim, CrossTransformerEncoderLayer(cli_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path))
            ]))

    def forward(self, derm_feat, cli_feat, meta_feat):
        derm_feat = derm_feat.unsqueeze(1)
        cli_feat = cli_feat.unsqueeze(1)
        meta_feat = self.proj_in(meta_feat).unsqueeze(1)

        # Fusion forawrd
        for meta_derm_fusion, meta_cli_fusion in self.layers:
            derm_feat = meta_derm_fusion(meta_feat, complement=derm_feat) + meta_feat
            cli_feat = meta_cli_fusion(meta_feat, complement=cli_feat) + meta_feat
        return derm_feat.squeeze(1), cli_feat.squeeze(1)