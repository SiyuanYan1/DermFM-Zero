# -*- coding： utf-8 -*-
import sys

import os

project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
models_path = os.path.join(project_root, 'models')

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)
sys.path.insert(0, models_path) 

from dataset.dataset import Derm7PtDatasetGroupInfrequent
from dataset.prompt import Derm7ptPromptBuilder, MILKPromptBuilder, PADPromptBuilder

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

#Build the Pytorch dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# dir_release = "/media/disk/zyl/data/derm7pt/release_v0/"
def load_derm7pt_meta(dir_release):
    dir_meta = os.path.join(dir_release,'meta')
    dir_images = os.path.join(dir_release,'images')

    meta_df = pd.read_csv(os.path.join(dir_meta,'meta.csv'))
    train_indexes = list(pd.read_csv(os.path.join(dir_meta,'train_indexes.csv'))['indexes'])
    valid_indexes = list(pd.read_csv(os.path.join(dir_meta,'valid_indexes.csv'))['indexes'])
    test_indexes = list(pd.read_csv(os.path.join(dir_meta,'test_indexes.csv'))['indexes'])

    # The dataset after grouping infrequent labels
    derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images,
                                                    metadata_df=meta_df.copy(),
                                                    train_indexes=train_indexes,
                                                    valid_indexes=valid_indexes,
                                                    test_indexes=test_indexes)

    # Print the details of dataset
    derm_data_group.dataset_stats()

    return derm_data_group

def process_multi_value_columns(df, columns):
    from sklearn.preprocessing import MultiLabelBinarizer
    
    processed_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # 处理缺失值和分割
            series = df[col].fillna('missing')
            split_values = series.apply(lambda x: [item.strip() for item in str(x).split(',')])
            
            # 使用MultiLabelBinarizer编码
            mlb = MultiLabelBinarizer()
            encoded_array = mlb.fit_transform(split_values)
            
            # 创建新列名并添加到DataFrame
            new_columns = [f"{col}_{class_}" for class_ in mlb.classes_]
            encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
            
            # 删除原列，添加新列
            processed_df = processed_df.drop(columns=[col])
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
    
    return processed_df

# mean and std for imagenet
mean = [0.485, 0.456, 0.406]
std = [0.228, 0.224, 0.225]

train_data_transformation = A.Compose(
    [   
        A.Resize(256,256),
        A.RandomResizedCrop(height=224, width=224),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ],
    additional_targets={'dermoscopy': 'image'})

test_data_transformation = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
], additional_targets={'dermoscopy': 'image'})

tta_data_transformations = [
    # Horizontal + Vertical flip
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # 90 degree rotation 
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.Rotate(limit=[90, 90], p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # 270 degree rotation
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.Rotate(limit=[270, 270], p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # Color jitter
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.HueSaturationValue(hue_shift_limit=64, sat_shift_limit=64, val_shift_limit=0, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # Brightness adjustment
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.RandomBrightness(limit=0.2, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # Contrast adjustment
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.RandomContrast(limit=0.2, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'}),
    
    # Gamma correction
    A.Compose([
        A.Resize(246, 246),
        A.CenterCrop(height=224, width=224),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'dermoscopy': 'image'})
]

def load_image(path, shape):

    img = cv2.imread(path)
    img = cv2.resize(img, (shape[0], shape[1]))

    return img

class Derm7ptDataset(Dataset):
    def __init__(self,derm,shape,mode='train', tokenizer=None):
        self.derm = derm
        self.shape = shape
        self.mode = mode
        self.derm_paths = derm.get_img_paths(data_type=mode, img_type='derm')
        self.clinic_paths = derm.get_img_paths(data_type=mode, img_type='clinic')
        self.labels = derm.get_labels(data_type=mode,one_hot=False) #dict list

        # self.meta -> one hot value
        # self.meta_text -> text with meta information
        if self.mode == 'train':
            self.meta = derm.meta_train.values
            self.meta_df = derm.meta_text_train
        elif self.mode == 'valid':
            self.meta = derm.meta_valid.values
            self.meta_df = derm.meta_text_valid
        else:
            self.meta = derm.meta_test.values
            self.meta_df = derm.meta_text_test

        # Convert meta into prompt(For text encoder)
        prompt_builder = Derm7ptPromptBuilder(self.meta_df)
        self.meta_text = [
            prompt_builder.build_patient_description(i) 
            for i in range(len(self.meta_df))]

        # This tokenizer is from our VL model
        self.tokenizer = tokenizer 

    def __getitem__(self, index):
        # get the dermoscopy image path
        dermoscopy_img_path = self.derm_paths[index]
        # get the clinic image path
        clinic_img_path = self.clinic_paths[index]
        # load the dermoscopy image
        dermoscopy_img = load_image(dermoscopy_img_path,self.shape)
        # load the clinic image
        clinic_img = load_image(clinic_img_path,self.shape)

        if self.mode == 'train':
            augmented = train_data_transformation(image=clinic_img, dermoscopy=dermoscopy_img)
        else:
            augmented = test_data_transformation(image=clinic_img, dermoscopy=dermoscopy_img)

        clinic_img = augmented['image']
        dermoscopy_img = augmented['dermoscopy']

        # label
        DIAG = torch.LongTensor([self.labels['DIAG'][index]])

        #meta data
        if self.tokenizer:
            metadata = self.tokenizer([str(self.meta_text[index])])[0]
        else: 
            metadata = torch.from_numpy(self.meta[index])

        return dermoscopy_img_path, clinic_img_path, dermoscopy_img, clinic_img, metadata,DIAG

    def __len__(self):
        return len(self.clinic_paths)

class MILK(Dataset):
    def __init__(self, df_path, shape, mode='train', tokenizer=None, MONET_feat=False):
        self.shape = shape
        self.mode = mode

        self.df = pd.read_csv(df_path).reset_index(drop=True)
        if MONET_feat:
            self.meta = pd.get_dummies(self.df[['age','sex', 'site', 'ulceration_crust' ,'vasculature_vessels','erythema','pigmented']])
        else:
            self.meta = pd.get_dummies(self.df[['age','sex', 'site']]) # meta input size: 10

        self.tokenizer = tokenizer

        if self.tokenizer:
            # Initialize prompt builder
            prompt_builder = MILKPromptBuilder(self.df)
            self.meta_text = [
                prompt_builder.build_patient_description(i, MONET_feat) 
                for i in range(len(self.df))
            ]
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # get the dermoscopy image path
        dermoscopy_img_path = row['derm_path']
        # get the clinic image path
        clinic_img_path = row['clinical_path']
        
        # load the dermoscopy image
        dermoscopy_img = load_image(dermoscopy_img_path, self.shape)
        # load the clinic image
        clinic_img = load_image(clinic_img_path, self.shape)
        
        if self.mode == 'train':
            augmented = train_data_transformation(image=clinic_img, dermoscopy=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['dermoscopy']
        elif self.mode == 'TTA':
            # Apply all TTA transformations and stack them
            clinic_img_tta = []
            dermoscopy_img_tta = []
            
            for tta_transform in tta_data_transformations:
                augmented = tta_transform(image=clinic_img, dermoscopy=dermoscopy_img)
                clinic_img_tta.append(augmented['image'])
                dermoscopy_img_tta.append(augmented['dermoscopy'])
            
            # Stack all TTA versions: [TTA_num, C, H, W]
            clinic_img = torch.stack(clinic_img_tta, dim=0)
            dermoscopy_img = torch.stack(dermoscopy_img_tta, dim=0) 
        else:
            augmented = test_data_transformation(image=clinic_img, dermoscopy=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['dermoscopy']

        if self.tokenizer:
            metadata = self.tokenizer([str(self.meta_text[index])])[0]
        else: 
            metadata = torch.from_numpy(self.meta.iloc[index].values.astype(np.float32))
        
        # label
        DIAG = torch.LongTensor([row['target']])
        
        return dermoscopy_img_path, clinic_img_path, dermoscopy_img, clinic_img, metadata, DIAG
    
    def __len__(self):
        return len(self.df)
    
class PAD(Dataset):
    _all_categories = None
    def __init__(self, df_path, shape, mode='train', tokenizer=None, fit_categories=False):
        self.shape = shape
        self.mode = mode

        self.df = pd.read_csv(df_path).reset_index(drop=True)

        self.tokenizer = tokenizer

        if self.tokenizer:
            # Initialize prompt builder
            prompt_builder = PADPromptBuilder(self.df)
            self.meta_text = [
                prompt_builder.build_patient_description(i) 
                for i in range(len(self.df))
            ]
        
        # Handle NA value with 0
        self.df = self.df.fillna(0)
        
        # Define columns for encoding
        self.meta_columns = ['smoke', 'drink', 'age', 'pesticide', 'gender', 'skin_cancer_history',
                            'cancer_history', 'fitspatrick', 'region', 
                            'diameter_1', 'diameter_2', 'itch', 'grew', 'hurt', 
                            'changed', 'bleed', 'elevation']
        
        # Create consistent one-hot encoded metadata
        if fit_categories or PAD._all_categories is None:
            # First time or explicitly fitting - store all categories
            self.meta = pd.get_dummies(self.df[self.meta_columns])
            PAD._all_categories = self.meta.columns.tolist()
            print(f"Fitted categories on {mode} set. Total features: {len(PAD._all_categories)}")
        else:
            # Use previously fitted categories
            self.meta = pd.get_dummies(self.df[self.meta_columns])
            
            # Align with fitted categories
            self.meta = self.meta.reindex(columns=PAD._all_categories, fill_value=0)
            print(f"Aligned {mode} set to fitted categories. Features: {len(PAD._all_categories)}")
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get the clinic image path
        clinic_img_path = row['clinical_path']
        
        # load the clinic image
        clinic_img = load_image(clinic_img_path, self.shape)
        
        if self.mode == 'train':
            augmented = train_data_transformation(image=clinic_img)
        else:
            augmented = test_data_transformation(image=clinic_img)
        
        clinic_img = augmented['image']

        if self.tokenizer:
            metadata = self.tokenizer([str(self.meta_text[index])])[0]
        else: 
            metadata = torch.from_numpy(self.meta.iloc[index].values.astype(np.float32))
        
        # label
        DIAG = torch.LongTensor([row['target']])
        
        return clinic_img_path, clinic_img_path, clinic_img, clinic_img, metadata, DIAG
    
    def __len__(self):
        return len(self.df)

    @classmethod
    def reset_categories(cls):
        """Reset the fitted categories (useful for new experiments)"""
        cls._all_categories = None