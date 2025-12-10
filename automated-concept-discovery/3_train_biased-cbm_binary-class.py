#!/usr/bin/env python3
"""
Simple script to train classifier on image embeddings with separate train/test CSVs.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import SGDClassifier

def main():
    parser = argparse.ArgumentParser(description='Train classifier on image embeddings')
    parser.add_argument('--train_csv', required=True, help='Path to training CSV file')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV file')
    parser.add_argument('--train_embeddings', required=True, help='Path to training embeddings .npy file')
    parser.add_argument('--test_embeddings', required=True, help='Path to test embeddings .npy file')
    parser.add_argument('--image_root', required=True, help='Path to image root')
    parser.add_argument('--image_col', required=True, help='image col')
    parser.add_argument('--output', type=str, default=None, help='Path to output folder')
    
    args = parser.parse_args()
    
    # Read train CSV
    print(f"Reading train CSV from {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    train_df[args.image_col] = train_df[args.image_col].apply(lambda x: os.path.join(args.image_root, x))
    
    # Read test CSV
    print(f"Reading test CSV from {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    test_df[args.image_col] = test_df[args.image_col].apply(lambda x: os.path.join(args.image_root, x))
    
    # Get label column (last column)
    label_col = train_df.columns[-1]
    print(f"Using label column: {label_col}")
    
    # Load embeddings
    print(f"Loading train embeddings from {args.train_embeddings}")
    X_train = np.load(args.train_embeddings)
    print(f"Train embeddings shape: {X_train.shape}")
    
    print(f"Loading test embeddings from {args.test_embeddings}")
    X_test = np.load(args.test_embeddings)
    print(f"Test embeddings shape: {X_test.shape}")
    
    # Prepare labels (using last column as target)
    y_train = train_df[label_col].values
    y_test = test_df[label_col].values
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    # Check if we have both classes in training data
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Training data only has {len(np.unique(y_train))} class(es). Need at least 2 classes.")
    
    # Train classifier
    print("Training classifier...")
    clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=0.001, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print(f"Macro F1 Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Save weights
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        np.save(os.path.join(args.output, 'classifier_weights.npy'), clf.coef_[0])
        np.save(os.path.join(args.output, 'classifier_intercept.npy'), clf.intercept_[0])
        print(f'Weights shape: {clf.coef_[0].shape}')
        print(f"\nSaved classifier weights to: {args.output}")
    
if __name__ == '__main__':
    main()