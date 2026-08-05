from itertools import repeat
import collections.abc
import logging

import torch
import torch.nn.functional as F
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def _interpolate_pos_embed(pos_embed, grid_h, grid_w, train_grid_h=14, train_grid_w=14):
    """Bicubic-interpolate the 2D position embedding (leading CLS slot) to a new
    patch grid. At 224 the grid is (14, 14) so this is an identity; it lets the
    wrapper transparently support other input resolutions.
    """
    if grid_h == train_grid_h and grid_w == train_grid_w:
        return pos_embed
    cls_pe = pos_embed[:, :1, :]
    patch_pe = pos_embed[:, 1:, :]
    D = patch_pe.shape[-1]
    patch_pe = patch_pe.reshape(1, train_grid_h, train_grid_w, D).permute(0, 3, 1, 2).float()
    patch_pe = F.interpolate(patch_pe, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, D)
    return torch.cat([cls_pe, patch_pe], dim=1)


class PanDermVisualWrapper(nn.Module):
    """DermFM-Zero vision tower: mean-pool the patch tokens, LayerNorm, then the
    image projection head.

    ``forward`` returns ``(pooled, patch_embeddings)``; ``CLIP.encode_image``
    keeps element [0]. Set ``use_prehead_features=True`` to expose the 1024-d
    pre-head pooled features (recommended for linear probing) instead of the
    768-d aligned projection.
    """

    def __init__(self, cae_model, image_size=224, use_prehead_features=False,
                 l2_normalize=False):
        super().__init__()
        self.trunk = cae_model
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.use_prehead_features = bool(use_prehead_features)
        self.l2_normalize = bool(l2_normalize)
        self.preprocess_cfg = {
            'size': self.image_size,
            'mode': 'RGB',
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'interpolation': 'bicubic',
            'resize_mode': 'shortest',
        }

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.trunk.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.trunk, 'set_grad_checkpointing'):
            self.trunk.set_grad_checkpointing(enable)

    def forward(self, x):
        patch_tokens, (grid_h, grid_w) = self.trunk.patch_embed(x, dynamic_size=True)
        batch_size = patch_tokens.size(0)

        cls_tokens = self.trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)

        if self.trunk.pos_embed is not None:
            train_grid_h, train_grid_w = self.trunk.patch_embed.patch_shape
            pos_embed = _interpolate_pos_embed(
                self.trunk.pos_embed, grid_h, grid_w, train_grid_h, train_grid_w,
            )
            x = x + pos_embed.expand(batch_size, -1, -1).type_as(x).to(x.device).detach()

        x = self.trunk.pos_drop(x)
        rel_pos_bias = self.trunk.rel_pos_bias() if self.trunk.rel_pos_bias is not None else None

        for blk in self.trunk.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        patch_tokens = x[:, 1:, :]
        if self.trunk.norm is not None:
            pooled = self.trunk.norm(patch_tokens.mean(1))
        else:
            pooled = x[:, 0]

        if self.use_prehead_features:
            if self.l2_normalize:
                pooled = F.normalize(pooled, dim=-1)
            return pooled, patch_tokens

        pooled = self.trunk.head(pooled)
        patch_embeddings = self.trunk.head(patch_tokens)
        if self.l2_normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled, patch_embeddings


def load_dermfm_checkpoint(model, ckpt_path, *, verbose=True):
    """Load a DermFM-Zero checkpoint into the CLIP model.

    Strips any DDP ``module.`` prefix, configures the text projection head to
    match the checkpoint layout (a BERT pooler + a 2-layer MLP projection), and
    loads the weights non-strictly so auxiliary keys are ignored.

    Returns ``(missing_keys, unexpected_keys)``.
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}

    _configure_text_head(model, sd)
    sd = {k: v for k, v in sd.items() if not k.startswith('knowledge_encoder.')}

    incompat = model.load_state_dict(sd, strict=False)
    if verbose:
        logging.info(
            f"[dermfm-zero] loaded {ckpt_path}: "
            f"missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}"
        )
    return incompat.missing_keys, incompat.unexpected_keys


# Backwards-compatible alias.
load_panderm_retrain_checkpoint = load_dermfm_checkpoint


def _configure_text_head(model, sd):
    """Set up the text pooler/projection to match the checkpoint layout."""
    if 'text.transformer.pooler.dense.weight' not in sd or not hasattr(model, 'text'):
        return
    from transformers.models.bert.modeling_bert import BertPooler
    from .hf_model import ClsPooler
    text_module = model.text
    bert = text_module.transformer
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if getattr(bert, 'pooler', None) is None:
        bert.pooler = BertPooler(bert.config).to(device=device, dtype=dtype)
    text_module.pooler = ClsPooler(use_pooler_output=True)
    embed_dim = sd['text.proj.0.weight'].shape[0]
    text_module.proj = nn.Sequential(
        nn.Linear(embed_dim, embed_dim, bias=True),
        nn.GELU(),
        nn.Linear(embed_dim, embed_dim, bias=True),
    ).to(device=device, dtype=dtype)


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype

def call_PanDerm_base_visual(vision_pretrain_path, linear_prob=False):
    from functools import partial

    if linear_prob:
        from CAE.models.modeling_finetune import VisionTransformer_LP as CAEVisionTransformer
        logging.info('Using Linear Probing Setting For PanDerm-Base')
    else:
        from CAE.models.modeling_finetune import VisionTransformer as CAEVisionTransformer
        logging.info('Using Default Setting For PanDerm-Base')

    kwargs = {
        'args': {
                'img_size': 224,                # From --model
                'patch_size': 16,               # From --model
                'in_chans': 3,                  # Standard for RGB images
                'embed_dim': 768,               # From --model (base model)
                'depth': 12,                    # Typical depth for base models
                'num_heads': 12,                # embed_dim / 64
                'mlp_ratio': 4.0,               # Common default
                'qkv_bias': True,               # As specified
                'norm_layer': partial(nn.LayerNorm, eps=1e-6),
                'init_values': 0.1,             # From --layer_scale_init_value
                'init_std': 0.02,               # Default value
                'drop_path_rate': 0.1,          # From --drop_path
                'decoder_embed_dim': 768,       # Same as embed_dim
                'decoder_num_classes': 8192,    # From --model
                'regressor_depth': 4,           # From --regressor_depth
                'decoder_depth': 4,             # From --decoder_depth
                'decoder_num_heads': 12,        # Same as num_heads
                'decoder_layer_scale_init_value': 0.1,  # From --decoder_layer_scale_init_value
                'fix_init_weight': False,       # As specified
                'model_type': 'caev2'
        }                     
    }

    model = CAEVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, init_values=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=512,  **kwargs)

    if vision_pretrain_path is not None:
        model.load_state_dict(torch.load(vision_pretrain_path), strict=False) 
        print(f'Successfully load panderm base vision encoder weight from {vision_pretrain_path}')
    return model

def call_PanDerm_large_visual(vision_pretrain_path, linear_prob, finetune):
    if linear_prob:
        from CAE.models.modeling_finetune import VisionTransformer_LP as CAEVisionTransformer
        print('Using Linear Probing Setting For PanDerm-Large')
    elif finetune:
        from CAE.models.modeling_finetune import VisionTransformer_FT as CAEVisionTransformer
        print('Using finetune Setting For PanDerm-Large')
    else:
        from CAE.models.modeling_finetune import VisionTransformer as CAEVisionTransformer
        print('Using default Setting For PanDerm-Large')

    from functools import partial
    def panderm_large_patch16_224(pretrained=False, **kwargs):
        model = CAEVisionTransformer(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1, num_classes=768, **kwargs)
        return model

    model = panderm_large_patch16_224()

    return model