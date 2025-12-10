#!/usr/bin/env python3
"""
Image Encoding Script using PanDerm Model

This script encodes images using a PanDerm OpenCLIP model and computes 
the mean embedding across all images. The embeddings are saved to disk 
for future use.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
import torch.nn.functional as F

class ImageDataset(Dataset):
    """Custom Dataset class for loading and preprocessing images"""
    
    def __init__(self, image_paths, preprocess):
        """
        Args:
            image_paths (list): List of image file paths
            preprocess: Image preprocessing transform
        """
        self.image_paths = image_paths
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single image
        
        Returns:
            tuple: (preprocessed_image, index)
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.preprocess(image)
        return image, idx


def encode_and_compute_mean(model, preprocess, df, batch_size=32, 
                           num_workers=16, device='cuda'):
    """
    Encode images in batches and compute the mean embedding
    
    Args:
        model: OpenCLIP model instance
        preprocess: Image preprocessing transform
        df (pd.DataFrame): DataFrame containing 'image_path' column
        batch_size (int): Number of images to process in each batch
        num_workers (int): Number of worker threads for data loading
        device (str): Device to use for computation ('cuda' or 'cpu')
    
    Returns:
        tuple: (all_embeddings, mean_embedding)
            - all_embeddings: numpy array of shape (N, embedding_dim)
            - mean_embedding: normalized mean embedding of shape (embedding_dim,)
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model components to appropriate devices
    # Keep text encoder on CPU to save GPU memory
    model.text = model.text.to('cpu')
    model.visual = model.visual.to(device)
    model.eval()
    
    # Create dataset and dataloader
    image_paths = df['image_path'].tolist()
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    all_embeddings = []
    
    print(f"Encoding {len(image_paths)} images...")
    
    # Encode all images
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            
            # Encode images using the visual encoder
            image_features = model.encode_image(images)
            
            # Normalize embeddings
            image_features = F.normalize(image_features, dim=1) # image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Move to CPU and convert to numpy
            all_embeddings.append(image_features.cpu().numpy())
    
    # Concatenate all batch embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Compute mean embedding across all images
    mean_embedding = np.mean(all_embeddings, axis=0)
    
    # Normalize the mean embedding
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    
    print(f"All embeddings shape: {all_embeddings.shape}")
    print(f"Mean embedding shape: {mean_embedding.shape}")
    
    return all_embeddings, mean_embedding

def load_model(model_name, checkpoint_path):
    """
    Load OpenCLIP model and checkpoint weights
    
    Args:
        model_name (str): Name of the OpenCLIP model architecture
        checkpoint_path (str): Path to the model checkpoint file
    
    Returns:
        tuple: (model, preprocess_transform)
    """
    print(f"Creating model: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=False)['state_dict']
    
        # Remove 'module.' prefix from keys (if exists from DataParallel training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
        model.load_state_dict(state_dict)

    print("Model loaded successfully")
    
    return model, preprocess

def load_and_prepare_dataframe(csv_path, data_root, img_col='filename'):
    """
    Load CSV file and prepare image paths
    
    Args:
        csv_path (str): Path to CSV file containing image filenames
        data_root (str): Root directory containing the images
    
    Returns:
        pd.DataFrame: DataFrame with 'image_path' column added
    """

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create full image paths by joining data_root with filenames
    df['image_path'] = df[img_col].apply(lambda x: os.path.join(data_root, x))
    
    print(f"Loaded {len(df)} image paths")
    
    # Verify that at least some images exist
    sample_paths = df['image_path'].head(10)
    existing = sum(os.path.exists(p) for p in sample_paths)
    print(f"Sample check: {existing}/10 images exist")
    
    return df


def save_embeddings(all_embeddings, output_dir):
    """
    Save embeddings to disk in multiple formats
    
    Args:
        all_embeddings (np.ndarray): All image embeddings
        mean_embedding (np.ndarray): Mean embedding
        output_dir (str): Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    npy_all = os.path.join(output_dir, 'all_embeddings.npy')
    np.save(npy_all, all_embeddings)
    print(f"Saved numpy arrays to: {npy_all}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Encode images using PanDerm OpenCLIP model'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_name',
        type=str,
        default='hf-hub:redlessone/PanDerm2',
        help='OpenCLIP model name'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    
    # Data parameters
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to CSV file containing image filenames'
    )
    parser.add_argument(
        '--img_col',
        type=str,
        default='filename',
        help='image filenames column'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing the images'
    )
    
    # Processing parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='Batch size for encoding (default: 2048)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of data loading workers (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./embeddings',
        help='Directory to save output embeddings (default: ./embeddings)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Load model and preprocessing transform
    model, preprocess = load_model(args.model_name, args.checkpoint)
    
    # Load and prepare dataframe with image paths
    df = load_and_prepare_dataframe(args.csv_path, args.data_root, args.img_col)
    
    # Encode images and compute mean embedding
    all_embeddings, mean_embedding = encode_and_compute_mean(
        model=model,
        preprocess=preprocess,
        df=df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )
    
    # Save embeddings to disk
    save_embeddings(all_embeddings, args.output_dir)
    
    print("\nEncoding complete!")
    print(f"Output files saved to: {args.output_dir}")

if __name__ == '__main__':
    main()