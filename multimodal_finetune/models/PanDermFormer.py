import warnings
warnings.filterwarnings('ignore')

import sys
import os

project_root = os.getcwd()
src_path = os.path.join(project_root, '../src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

import torch
import torch.nn as nn
from CAE.models.modeling_finetune import panderm_base_patch16_224, panderm_large_patch16_224
from models.mca import CrossTransformer_meta
from models.fusion import CrossTransformer
import torch.nn.functional as F
import open_clip
from models.transformer import create_vit_l14_text_encoder, nn_Lambda
from transformers import AutoModelForZeroShotImageClassification

class MLP(nn.Module):
    '''
    MLP that is used for the subnetwork of metadata.
    '''
    def __init__(self,in_size,hidden_size,out_size,dropout=0.5):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        '''
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)

        return x

class MetaSubNet(nn.Module):
    '''
    The subnetwork that is used for metadata.
    Maybe the subnetwork that is not needed in the task
    '''
    def __init__(self, in_size, hidden_size, out_size, dropout=0.2):
        '''
        :param in_size: input dimension
        :param hidden_size: hidden layer dimension
        :param out_size: output dimension
        :param dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(MetaSubNet, self).__init__()
        self.rnn = MLP(in_size,hidden_size,out_size,dropout=dropout)

    def forward(self, x):
        '''
        :param x: tensor of shape (batch_size, sequence_len, in_size)
        :return: tensor of shape (batch_size,out_size)
        '''
        meta_output = self.rnn(x)
        return meta_output

def load_open_clip_state_dict(model, pretrain_path):
    # Load panderm large
    state_dict = torch.load(pretrain_path, weights_only=False)['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

class PanDermTFormer(nn.Module):
    def __init__(self,
                 model_name='PanDerm-Base', 
                 num_classes=5, 
                 pretrain_path=None, 
                 hidden_dim=768, 
                 use_cli=True, 
                 use_derm=True, 
                 use_meta=True,
                 use_text_encoder=False, 
                 fusion_mode='concatenate', 
                 meta_fusion_mode='concatenate',
                 out='linear', 
                 encoder_pool='mean', 
                 use_visual_embedding_layer=False,
                 num_head=8, 
                 att_depth=4,
                 meta_num_head=8,
                 meta_att_depth=2,
                 meta_dim=128,
                 meta_input_dim=14):

        super(PanDermTFormer, self).__init__()
        # Store which modalities to use
        self.use_cli = use_cli
        self.use_derm = use_derm
        self.use_meta = use_meta
        self.fusion_mode = fusion_mode
        self.meta_fusion_mode = meta_fusion_mode
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        self.out = out
        self.use_text_encoder = use_text_encoder
        self.model_name = model_name
        self.use_visual_embedding_layer = use_visual_embedding_layer
        
        # Initialize encoders
        self.build_encoder(model_name, pretrain_path, encoder_pool)

        # Create fusion cross attention
        if self.use_cli and self.use_derm and self.fusion_mode == 'cross attention':
            self.fusion = CrossTransformer(x_dim=self.hidden_dim,c_dim=self.hidden_dim, depth=att_depth, num_heads=num_head)
            print('Visual modality fusion: cross attention')
        else:
            print('Visual modality fusion: concatenate')

        # Visual embedding is used to help the model distinguish between the two modalities
        if self.use_visual_embedding_layer:
            self.cli_visual_embedding_layer = nn.Linear(self.hidden_dim, self.hidden_dim) if self.use_cli else None
            self.derm_visual_embedding_layer = nn.Linear(self.hidden_dim, self.hidden_dim) if self.use_derm else None

        if self.use_meta:
            if self.use_text_encoder:
                self.build_text_encoder(model_name, pretrain_path)
                self.meta_proj = nn.Linear(meta_dim, self.hidden_dim)
                meta_in_dim = self.hidden_dim

            else:
                self.meta_encoder = MetaSubNet(meta_input_dim, self.meta_dim, self.meta_dim, 0.3)
                self.meta_proj = nn.Identity()
                meta_in_dim = self.meta_dim

            if self.meta_fusion_mode == 'cross attention':
                self.meta_fusion = CrossTransformer_meta(meta_in_dim, self.hidden_dim*(int(self.use_cli) + int(self.use_derm)), depth=meta_att_depth, num_heads=meta_num_head)
                print(f"Enable meta fusion using cross transformer with {meta_num_head} heads, {meta_att_depth} depth")
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.num_classes = num_classes
        
        # Calculate final dimension dynamically
        final_dim = self._calculate_final_dim()
        
        if self.out == 'linear':
            self.head = nn.Linear(final_dim, self.num_classes)
        elif self.out == 'mlp':
            self.head = nn.Sequential(
                           nn.Linear(final_dim, 256),
                           nn.BatchNorm1d(256),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.3),
                           nn.Linear(256, self.num_classes)
                           )
        print(f"Out layer: {out}")
        
        if pretrain_path is not None:
            print(f"Load pretrain checkpoint from {pretrain_path}")

    def build_encoder(self, model_name, pretrain_path, encoder_pool):
        """Build vision encoders for clinical and dermascopic images"""

        def create_vision_encoder(model_name, encoder_pool):
            if encoder_pool == 'mean':
                if model_name == 'PanDerm-Base':
                    encoder = panderm_base_patch16_224(lin_probe=False, use_mean_pooling=True)
                elif model_name == 'PanDerm-Large':
                    encoder = panderm_large_patch16_224(lin_probe=False, use_mean_pooling=True)
                    encoder.head = nn.Identity()
                elif model_name == 'PanDerm-Large-VL':
                    encoder = open_clip.create_model_and_transforms('PanDerm-large-w-PubMed-256')[0]
                    if pretrain_path:
                        load_open_clip_state_dict(encoder, pretrain_path)
                    encoder = encoder.visual
                elif model_name == 'PanDerm-v2':
                    encoder = open_clip.create_model_and_transforms('hf-hub:redlessone/PanDerm2', finetune=True)[0]
                    print("Calling Pandermv2 checkpoint from huggingface api")
                    encoder = encoder.visual
                elif model_name == 'BioMedCLIP':
                    encoder = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')[0]
                    encoder = encoder.visual
                elif model_name == 'CLIP-L14':
                    encoder = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')[0]
                    encoder = encoder.visual
                    encoder.proj = None
                elif model_name == 'MONET':
                    encoder = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
                    encoder = nn.Sequential(encoder.vision_model, nn_Lambda(lambda x: x['pooler_output']))
                elif model_name == 'ViT-Large':
                    import timm
                    encoder = timm.create_model('timm/vit_large_patch16_224.augreg_in21k_ft_in1k',
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  pretrained=True)
                elif model_name == 'DINOv2':
                    import timm
                    encoder = timm.create_model("vit_large_patch14_dinov2.lvd142m",
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  pretrained=True)
                    
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

            elif encoder_pool == 'cls':
                if model_name == 'PanDerm-Base':
                    encoder = panderm_base_patch16_224(lin_probe=False, use_mean_pooling=False)
                else:
                    raise ValueError(f"CLS pooling only supported for PanDerm-Base, got {model_name}")
            else:
                raise ValueError(f"Unsupported pooling method: {encoder_pool}")

            # For PanDerm-Large and PanDerm-Base
            if pretrain_path and model_name not in ['PanDerm-Large-VL', 'PanDerm-v2','BioMedCLIP', 'CLIP-L14']:
                encoder.load_state_dict(torch.load(pretrain_path, map_location='cpu')['state_dict'], strict=False)

            # Remove classification head - Replace it with identity layer
            encoder.head = nn.Identity()
            return encoder

        def log_encoder_initialization(encoder_type, model_name, pretrain_path):
            if pretrain_path:
                print(f"Initialize {encoder_type} vision encoder with {model_name} weight: {pretrain_path}")

        # Build clinical image encoder
        if self.use_cli:
            self.cli_vit = create_vision_encoder(model_name, encoder_pool)
            log_encoder_initialization("clinical", model_name, pretrain_path)

        # Build dermascopic image encoder  
        if self.use_derm:
            self.derm_vit = create_vision_encoder(model_name, encoder_pool)
            log_encoder_initialization("dermascopic", model_name, pretrain_path)
    
    def build_text_encoder(self, model_name, pretrain_path):
        if model_name == 'PanDerm-Large-VL':
            encoder = open_clip.create_model_and_transforms('PanDerm-large-w-PubMed-256')[0]
            if pretrain_path is not None:
                load_open_clip_state_dict(encoder, pretrain_path)
                print(f"Initialize text encoder with PanDerm-Large-VL weight: {pretrain_path}")
            self.meta_encoder = encoder.text
        elif model_name == 'PanDerm-v2':
            encoder = open_clip.create_model_and_transforms('hf-hub:redlessone/PanDerm2')[0]
            self.meta_encoder = encoder.text
        elif model_name == 'BioMedCLIP':
            encoder = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')[0]
            self.meta_encoder = encoder.text
        elif model_name == 'CLIP-L14' or model_name == 'PanDerm-Large':
            self.meta_encoder = create_vit_l14_text_encoder()
        elif model_name == 'MONET':
            encoder = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
            self.meta_encoder = nn.Sequential(encoder.text_model, nn_Lambda(lambda x: x['pooler_output']), encoder.text_projection)
    
    def _calculate_final_dim(self):
        """Calculate the final feature dimension"""
        if self.use_meta and (self.use_cli and self.use_derm):
            # Meta fusion output + one visual modality
            dim = self.meta_dim + self.hidden_dim * (int(self.use_cli) + int(self.use_derm)) if self.meta_fusion_mode != 'cross attention' or not self.use_text_encoder else self.hidden_dim * (int(self.use_cli) + int(self.use_derm) + int(self.use_meta))
        elif self.use_cli and self.use_derm and not self.use_meta:
            # Both visual modalities without meta
            dim = self.hidden_dim * 2

        # Single visual Dim and meta dim
        elif (self.use_cli or self.use_derm) and self.use_meta:
            dim = self.hidden_dim + self.meta_dim if not self.use_text_encoder else self.hidden_dim*2

        # Single visiual dim
        elif (self.use_cli or self.use_derm) and not self.use_meta:
            dim = self.hidden_dim
        
        # Only text encoder
        else:
            dim = self.hidden_dim
        
        print(f"Final head dim: {dim}")
        return dim
    
    # import torchsnooper
    # @torchsnooper.snoop()
    def forward(self, meta_x=None, cli_x=None, der_x=None):
        features = []
        
        # Extract features from each modality
        if self.use_meta:
            if meta_x is None:
                raise ValueError("meta_x required when use_meta=True")
            meta_h = self.meta_encoder(meta_x)
            meta_h = self.meta_proj(meta_h)
            features.append(meta_h)
            
        if self.use_cli:
            if cli_x is None:
                raise ValueError("cli_x required when use_cli=True")
            cli_f = self.cli_vit(cli_x)

            cli_f = self.cli_visual_embedding_layer(cli_f) if self.use_visual_embedding_layer else cli_f

            features.append(cli_f)
            
        if self.use_derm:
            if der_x is None:
                raise ValueError("der_x required when use_derm=True")
            der_f = self.derm_vit(der_x)

            der_f = self.derm_visual_embedding_layer(der_f) if self.use_visual_embedding_layer else der_f

            features.append(der_f)
        
        # enable vision fusion and meta fusion
        if self.use_cli and self.use_derm and self.fusion_mode == 'cross attention':
            cli_f_derm, der_f_cli = self.fusion(cli_f, der_f)

            # Using meta fusion - We use meta as query and use clinical,derm feature as key and value.
            if self.use_meta and self.meta_fusion_mode == 'cross attention':
                feature_f = torch.cat([cli_f,der_f],dim=-1)
                x = self.meta_fusion(meta_h, feature_f)
                features = [der_f_cli, x, cli_f_derm]
            else:
                features = [cli_f_derm, der_f_cli] if self.use_cli and self.use_derm and not self.use_meta else [meta_h, cli_f_derm, der_f_cli]
        
        # enable single vision modality and meta fusion
        elif self.use_cli:
            if self.use_meta and self.meta_fusion_mode == 'cross attention':
                x = self.meta_fusion(meta_h, cli_f)
                features = [cli_f, x]
            elif not self.use_meta:
                features = [cli_f] 
            else:
                raise NotImplementedError

        # enable single vision modality and meta fusion
        elif self.use_derm:
            if self.use_meta and self.meta_fusion_mode == 'cross attention':
                x = self.meta_fusion(meta_h, der_f)
                features = [der_f, x]
            elif not self.use_meta:
                features = [der_f] 
            else:
                raise NotImplementedError
        elif self.use_meta and self.use_text_encoder:
            features = [meta_h]
        else:
            raise NotImplementedError

        x = torch.cat(features, dim=-1)

        # Classification
        diag = self.head(x)
        return diag
    
    def criterion(self, logit, truth, weight=None, multi_label=False):
        # This loss is used for multi-label task
        if multi_label:
            logit = torch.sigmoid(logit)
            return F.binary_cross_entropy(logit, truth)
        
        # Default use
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth, weight=weight)
        return loss