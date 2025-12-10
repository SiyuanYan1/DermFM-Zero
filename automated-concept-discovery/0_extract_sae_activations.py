#!/usr/bin/env python3
"""
Simple script to extract learned activations using Sparse Autoencoder.
"""
import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append('../sparse_autoencoder/')
from sae.sparse_autoencoder import (
    SparseAutoencoder,
)

def load_sae_model(checkpoint_path, device='cuda'):
    """Load Sparse Autoencoder model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    
    autoencoder = SparseAutoencoder(
        n_input_features=config['n_input_features'],
        n_learned_features=config['n_learned_features'],
        n_components=config['n_components']
    ).to(device)
    
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Model config: {config}")
    
    return autoencoder


def extract_activations(autoencoder, embeddings, batch_size=512, device='cuda'):
    """Extract learned activations from embeddings."""
    embeddings = torch.from_numpy(embeddings)
    learned_activations = []
    indices = []
    
    n_batches = (len(embeddings) + batch_size - 1) // batch_size
    print(f"Processing {len(embeddings)} embeddings in {n_batches} batches...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(embeddings))
            
            batch = embeddings[start_idx:end_idx].to(device)
            concept_batch = autoencoder(batch)
            
            # Extract learned activations
            learned_act_batch = concept_batch.learned_activations[:, 0, :].cpu()
            learned_activations.append(learned_act_batch)
            
            # Track active indices
            index_batch = torch.abs(concept_batch.learned_activations).sum(dim=[0, 1]).cpu() > 0
            indices.append(index_batch.unsqueeze(0))
    
    learned_activations = torch.cat(learned_activations, dim=0)
    indices = torch.cat(indices, dim=0)
    active_indices = indices.sum(dim=0) > 0
    
    print(f"Output shape: {learned_activations.shape}")
    print(f"Active features: {active_indices.sum().item()} / {len(active_indices)}")
    
    return learned_activations.numpy(), active_indices.numpy()


def main():
    parser = argparse.ArgumentParser(description='Extract learned activations using SAE')
    parser.add_argument('--checkpoint', required=True, help='Path to SAE checkpoint (.pth)')
    parser.add_argument('--embeddings', required=True, help='Path to input embeddings (.npy)')
    parser.add_argument('--output', required=True, help='Output path for learned activations (.npy)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load SAE model
    print("\nLoading SAE model...")
    autoencoder = load_sae_model(args.checkpoint, device=device)
    
    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Extract learned activations
    print("\nExtracting learned activations...")
    learned_activations, _ = extract_activations(
        autoencoder, embeddings, batch_size=args.batch_size, device=device
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, learned_activations)
    print(f"\nLearned activations saved to: {args.output}")

if __name__ == '__main__':
    main()