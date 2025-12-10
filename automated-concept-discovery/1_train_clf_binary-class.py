#!/usr/bin/env python3
"""
Simple script to train XGBoost classifier on image embeddings.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on image embeddings')
    parser.add_argument('--csv', required=True, help='Path to CSV metadata file')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings .npy file')
    parser.add_argument('--image_col', default='ImageID', help='Column name for image IDs')
    parser.add_argument('--image_dir', help='Image directory (optional, for building paths)')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument('--output', type=str, default=None, help='Path to output folder')
    
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
    
    # Prepare labels (use argmax for multi-class)
    Y = gt_results[:,1]
    print(f"Shape of Y: {Y.shape}")
    
    # Train multiple times
    roc_auc_scores = []
    f1_scores = []

    all_weights = []
    all_intercepts = []

    from sklearn.linear_model import SGDClassifier

    print("Start training linear model")
    for i in tqdm(range(50)):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=True, random_state=i, stratify=Y)

        clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001)
        clf.fit(X_train, y_train)

        clf.fit(X_train, y_train)
    
        # Save concept weight
        all_weights.append(clf.coef_[0])
        all_intercepts.append(clf.intercept_[0])

        # Calculate ROC AUC score and add it to the list
        score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        roc_auc_scores.append(score)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

    # Calculate the mean ROC AUC score
    mean_roc_auc = np.mean(roc_auc_scores)
    mean_f1 = np.mean(f1_scores)

    mean_weights = np.mean(all_weights, axis=0)
    mean_intercept = np.mean(all_intercepts)

    if args.output:
        np.save(os.path.join(args.output, 'mean_classifier_weights.npy'), mean_weights)
        np.save(os.path.join(args.output, 'mean_classifier_intercept.npy'), mean_intercept)
        print(f"Save mean CBM classifier weight to output folder: {args.output}")

    print(f"Mean ROC AUC Score over 50 iterations: {mean_roc_auc:.3f}")
    print(f"Mean Macro F1 Score over 50 iterations: {mean_f1:.3f}")
    
if __name__ == '__main__':
    main()