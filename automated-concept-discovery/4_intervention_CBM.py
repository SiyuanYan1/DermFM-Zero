#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import random
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

# Add src path for importing open_clip
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from src import open_clip

def load_model_and_embeddings(sae_path, clip_model_name, clip_checkpoint_path, 
                             device='cpu'):
    """
    Load SAE dictionary vectors and CLIP model.
    
    Args:
        sae_path: Path to the SAE autoencoder checkpoint
        clip_model_name: Name of the CLIP model architecture
        clip_checkpoint_path: Path to the CLIP model checkpoint
        device: Device to load models on ('cpu' or 'cuda')
    
    Returns:
        dict_vec: Dictionary vectors from SAE decoder [feature_dim, num_neurons]
        model: Loaded CLIP model
        tokenizer: CLIP tokenizer
    """
    # Load SAE dictionary vectors
    print(f"Loading SAE checkpoint from: {sae_path}")
    sae_checkpoint = torch.load(sae_path, map_location=device)
    dict_vec = sae_checkpoint['model_state_dict']['decoder._weight'].to(device).squeeze()
    print(f"Dictionary vectors shape: {dict_vec.shape}")
    
    # Load CLIP model
    print(f"Loading CLIP model: {clip_model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    
    # Load checkpoint
    if clip_checkpoint_path:
        print(f"Loading CLIP checkpoint from: {clip_checkpoint_path}")
        state_dict = torch.load(clip_checkpoint_path, weights_only=False, map_location=device)['state_dict']
    
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.to(device)
    print("Successfully loaded CLIP model")
    
    return dict_vec, model, tokenizer


def get_concept_embeddings(model, tokenizer, concept_list, device='cpu'):
    """
    Get normalized embeddings for concept list.
    
    Args:
        model: CLIP model
        tokenizer: CLIP tokenizer
        concept_list: List of concept strings
        device: Device to compute on
    
    Returns:
        concept_embeddings_norm: Normalized concept embeddings [num_concepts, feature_dim]
    """
    concept_embeddings = model.encode_text(tokenizer(concept_list).to(device))
    concept_embeddings_norm = F.normalize(concept_embeddings, dim=1)
    return concept_embeddings_norm


def optimal_concept_matching(dict_vec, concept_embeddings_norm, importances, 
                            concept_list, device='cpu'):
    """
    Use Hungarian algorithm to find globally optimal concept-to-neuron matching.
    
    This function matches each concept to its most similar neuron by:
    1. Computing cosine similarity between concept embeddings and neuron dictionary vectors
    2. Using the Hungarian algorithm to find optimal one-to-one matching
    3. Ranking results by neuron importance
    
    Args:
        dict_vec: Dictionary vectors (neurons) [feature_dim, num_neurons]
        concept_embeddings_norm: Normalized concept embeddings [num_concepts, feature_dim]
        importances: Importance scores for each neuron [num_neurons]
        concept_list: List of concept text descriptions
        device: Computing device
    
    Returns:
        matched_concepts: List of matching results sorted by concept order
        sorted_concepts: Matching results sorted by importance (descending)
    """
    # Calculate importance ranks (higher importance = lower rank number)
    importance_ranks = np.argsort(-np.abs(importances))  # Descending order by absolute value
    rank_map = {neuron_idx: rank for rank, neuron_idx in enumerate(importance_ranks)}
    
    # Filter out neurons with zero importance to speed up matching
    non_zero_mask = importances != 0
    non_zero_indices = np.where(non_zero_mask)[0]
    
    print(f"Total neurons: {len(importances)}")
    print(f"Non-zero importance neurons: {len(non_zero_indices)}")
    
    # Extract only non-zero importance neurons
    dict_vec_filtered = dict_vec[:, non_zero_indices]
    
    # # Normalize dictionary vectors for cosine similarity
    # dict_vec_norm = F.normalize(dict_vec_filtered, dim=0)
    
    # Compute similarity matrix [num_concepts, num_neurons]
    similarity = concept_embeddings_norm @ dict_vec_filtered
    
    # Convert to cost matrix for Hungarian algorithm (negative similarity = cost)
    cost_matrix = -similarity.detach().cpu().numpy()
    
    # Use Hungarian algorithm to find optimal one-to-one matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build matching results with metadata
    matched_concepts = []
    for concept_idx, filtered_neuron_idx in zip(row_ind, col_ind):
        # Map back to original neuron index (before filtering)
        original_neuron_idx = non_zero_indices[filtered_neuron_idx]
        
        matched_concepts.append({
            'concept_idx': concept_idx,
            'concept': concept_list[concept_idx],
            'neuron_idx': original_neuron_idx,
            'similarity': similarity[concept_idx, filtered_neuron_idx].item(),
            'importance': importances[original_neuron_idx],
            'importance_rank': rank_map[original_neuron_idx]
        })
    
    # Sort by concept index (maintains input order)
    matched_concepts.sort(key=lambda x: x['concept_idx'])
    
    # Sort by absolute importance (descending)
    sorted_concepts = sorted(matched_concepts, key=lambda x: abs(x['importance']), reverse=True)
    
    return matched_concepts, sorted_concepts


def select_topk_neurons(importances, k=10):
    """
    Select top-k neurons by absolute importance.
    
    Args:
        importances: Importance scores for each neuron [num_neurons]
        k: Number of top important neurons to select
    
    Returns:
        topk_indices: Indices of top-k neurons
        topk_importances: Importance values of top-k neurons
    """
    # Filter out neurons with zero importance
    non_zero_mask = importances != 0
    non_zero_indices = np.where(non_zero_mask)[0]
    non_zero_importances = importances[non_zero_indices]
    
    print(f"Total neurons: {len(importances)}")
    print(f"Non-zero importance neurons: {len(non_zero_indices)}")
    
    # Select top-k neurons by absolute importance value
    topk = min(k, len(non_zero_indices))
    topk_local_indices = np.argsort(np.abs(non_zero_importances))[-topk:][::-1]
    topk_indices = non_zero_indices[topk_local_indices]
    topk_importances = importances[topk_indices]
    
    print(f"Selected top-{topk} neurons by importance")
    
    return topk_indices, topk_importances


def intervene_activations(activation_path, neuron_indices, output_path):
    """
    Set specified neurons to zero in activations.
    
    Args:
        activation_path: Path to original activation file (.npy)
        neuron_indices: List of neuron indices to set to zero
        output_path: Path to save intervened activations
    
    Returns:
        activations_masked: Intervened activation array
    """
    print(f"\nLoading activations from: {activation_path}")
    activations = np.load(activation_path)
    print(f"Activation shape: {activations.shape}")
    
    # Copy original activations and mask specified neurons
    activations_masked = activations.copy()
    activations_masked[:, neuron_indices] = 0
    
    print(f"Set {len(neuron_indices)} neurons to zero: {neuron_indices}")
    
    # Save intervened activations
    np.save(output_path, activations_masked)
    print(f"Saved intervened activations to: {output_path}")
    
    return activations_masked


def main():
    parser = argparse.ArgumentParser(
        description='Identify and intervene top-N neurons related to specific concepts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === Core Parameters ===
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top neurons to intervene on')
    parser.add_argument('--concept', type=str, default='hair',
                       help='Concept to search for (used in matching method)')
    parser.add_argument('--method', type=str, default='matching', 
                       choices=['matching', 'topk'],
                       help='Method to select neurons: "matching" (concept matching) or "topk" (by importance)')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'test'],
                       help='Data split to intervene on')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    # === Path Parameters ===
    parser.add_argument('--data_dir', type=str, 
                       default='/mnt/hdd/sda/xjli/repos/MAKE/data/ISIC_hair_bias-XJ',
                       help='Data directory containing images and metadata')
    parser.add_argument('--result_dir', type=str,
                       default='/mnt/hdd/sda/xjli/repos/CFN/result/ISIC_hair_bias-XJ/PanDermv2',
                       help='Result directory containing activations and importance weights')
    parser.add_argument('--project_root', type=str, default='../',
                       help='Project root directory (for importing modules)')
    
    # === Model Parameters ===
    parser.add_argument('--sae_checkpoint', type=str,
                       default='/mnt/hdd/sda/xjli/repos/CFN/result-pretrain-using-pandermv2-data/pandermv2-pretrain/PanDermv2/autoencoder.pth',
                       help='Path to SAE autoencoder checkpoint')
    parser.add_argument('--clip_model_name', type=str,
                       default='PanDerm-large-w-PubMed-256',
                       help='Name of the CLIP model architecture')
    parser.add_argument('--clip_checkpoint', type=str,
                       default=None,
                       help='Path to CLIP model checkpoint')
    
    # === File Name Parameters ===
    parser.add_argument('--importance_file', type=str,
                       default='before-intervention/classifier_weights.npy',
                       help='Relative path to importance weights file within result_dir')
    parser.add_argument('--activation_file', type=str,
                       default='learned_activation.npy',
                       help='Name of the activation file within split directory')
    parser.add_argument('--output_file', type=str,
                       default='learned_activation_after_intervention.npy',
                       help='Name of the output file for intervened activations')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("Neuron Intervention Script")
    print("="*80)
    print(f"Top-N: {args.top_n}")
    print(f"Concept: {args.concept}")
    print(f"Method: {args.method}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Load importance weights
    importance_path = os.path.join(args.result_dir, args.importance_file)
    print(f"Loading importance weights from: {importance_path}")
    importances = np.load(importance_path)
    print(f"Importance shape: {importances.shape}\n")
    
    if args.method == 'matching':
        # Method 1: Concept matching using semantic similarity
        print("="*80)
        print("Method: Concept Matching")
        print("="*80)
        
        # Load model and get concept embeddings
        dict_vec, model, tokenizer = load_model_and_embeddings(
            args.sae_checkpoint, 
            args.clip_model_name, 
            args.clip_checkpoint,
            args.device
        )
        
        # Create concept list (repeat concept for multiple matching)
        concept_list = [args.concept] * args.top_n
        print(f"\nConcept list: {concept_list}")
        
        concept_embeddings_norm = get_concept_embeddings(
            model, tokenizer, concept_list, args.device
        )
        print(f"Concept embeddings shape: {concept_embeddings_norm.shape}\n")
        
        # Find optimal matching using Hungarian algorithm
        matched, sorted_matched = optimal_concept_matching(
            dict_vec, concept_embeddings_norm, importances, concept_list, args.device
        )
        
        # Print matching results
        print("\n" + "="*80)
        print("Top-N Matched Concepts (sorted by importance)")
        print("="*80)
        for i, item in enumerate(sorted_matched, 1):
            print(f"{i:2d}. Concept: {item['concept']:20s} -> Neuron {item['neuron_idx']:4d} | "
                  f"Rank: {item['importance_rank']:4d}/{len(importances)} | "
                  f"Importance: {item['importance']:+.6f} | Similarity: {item['similarity']:.4f}")
        
        # Extract neuron indices
        neuron_indices = [item['neuron_idx'] for item in sorted_matched]
        
    else:
        # Method 2: Top-K by importance (no concept matching)
        print("="*80)
        print("Method: Top-K Importance")
        print("="*80)
        
        topk_indices, topk_importances = select_topk_neurons(importances, k=args.top_n)
        
        print("\n" + "="*80)
        print(f"Top-{args.top_n} Neurons by Importance")
        print("="*80)
        for rank, (idx, imp) in enumerate(zip(topk_indices, topk_importances), 1):
            print(f"Rank {rank:2d}: Neuron {idx:4d} | Importance: {imp:+.6f}")
        
        neuron_indices = topk_indices.tolist()
    
    # Perform intervention on activations
    print("\n" + "="*80)
    print("Performing Intervention")
    print("="*80)
    
    activation_path = os.path.join(args.result_dir, args.split, args.activation_file)
    output_path = os.path.join(args.result_dir, args.split, args.output_file)
    
    intervene_activations(activation_path, neuron_indices, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("Intervention Complete!")
    print("="*80)
    print(f"Number of intervened neurons: {len(neuron_indices)}")
    print(f"Intervened neuron indices: {neuron_indices}")
    print(f"Output saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()