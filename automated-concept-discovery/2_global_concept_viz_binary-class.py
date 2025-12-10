#!/usr/bin/env python3
"""
Simple script to train XGBoost classifier on image embeddings.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

class AverageClassifier:
        def __init__(self, importances):
            self.importances = [importances]

def print_concept_weights(concept_presence_score_dict, clf):
    """print concept weights"""
    concept_names = list(concept_presence_score_dict.keys())
    weights = clf.importances[0]
    
    for i, concept_name in enumerate(concept_names):
        print(f"{concept_name}: {weights[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on image embeddings')
    parser.add_argument('--csv', required=True, help='Path to CSV metadata file')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings .npy file')
    parser.add_argument('--image_col', default='ImageID', help='Column name for image IDs')
    parser.add_argument('--image_dir', help='Image directory (optional, for building paths)')
    parser.add_argument('--out_dir', help='output directory')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument('--topk', type=int, default=10, help='top k features per class')  # 新增
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Read CSV
    print(f"Reading CSV from {args.csv}")
    df = pd.read_csv(args.csv)
    
    # Build image paths if needed
    if args.image_dir:
        df['image_path'] = df[args.image_col].apply(
            lambda x: os.path.join(args.image_dir, x)
        )
    
    # Build ground truth matrix
    label_cols = df.columns[1:-1].tolist() if 'image_path' in df.columns else df.columns[1:].tolist()
    
    print(f"Using label columns: {label_cols}")
    gt_results = df[label_cols].values
    print(f"Ground truth shape: {gt_results.shape}")
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    X = np.load(args.embeddings)
    print(f"Embeddings shape: {X.shape}")
    
    # Prepare labels (use argmax for binary class)
    Y = gt_results[:,1]
    print(f"Shape of Y: {Y.shape}")
    
    # ============ class-specific importance ============
    print("\n" + "="*60)
    print("Computing class-specific feature importance...")
    
    # Center features
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    unique_classes = np.unique(Y)
    class_importance = {}
    
    for class_idx in unique_classes:
        # Get samples for this class
        class_mask = (Y == class_idx)
        class_X = X_centered[class_mask]
        print(f"Class {class_idx}: {class_X.shape}")
        
        # Mean activation per feature for this class
        mean_activations = class_X.mean(axis=0)
        
        # Get top-k features
        top_indices = np.argsort(np.abs(mean_activations))[-args.topk:][::-1]
        # top_indices = np.argsort(mean_activations)[-args.topk:][::-1]

        top_values = mean_activations[top_indices]
        
        class_importance[int(class_idx)] = {
            'top_indices': top_indices,
            'top_values': top_values,
            'mean_activations': mean_activations
        }
        
        print(f"Class {int(class_idx)} - (activation: {mean_activations[top_indices]}): \nTop {args.topk} features = {top_indices.tolist()}")
    
    # Save class importance
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'class_importance.npy'), class_importance)
    print(f"\nClass importance saved to {os.path.join(args.out_dir, 'class_importance.npy')}")
    
    
if __name__ == '__main__':
    main()