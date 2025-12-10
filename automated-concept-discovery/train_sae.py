import numpy as np
import torch
import os.path as osp
from sae.custom_pipeline import Pipeline
import os
from pathlib import Path
import torch
import numpy as np
import math
import sys

sys.path.append('./sparse_autoencoder/')
from sae.sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
from time import time

import os.path as osp
import argparse

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    parser.add_argument('--expansion', default=8)
    parser.add_argument('--backbone', default=None)
    parser.add_argument('--l1', default=3e-5)
    parser.add_argument('--trial', default="")
    parser.add_argument('--save_dir', default="")
    args = parser.parse_args()

    data_dir_activations = {}
    data_dir_activations["img"] = os.path.join(args.save_dir, 'temp_out')

    # os.mkdir(args.save_dir)
    # os.mkdir(data_dir_activations["img"])
    
    save_dir_activations = args.save_dir
    train = torch.from_numpy(np.load(save_dir_activations + "all_embeddings.npy"))
    torch.save(train, osp.join(data_dir_activations["img"], "train"))

    trian_val = None
    val = None

    start_time = time()
    autoencoder_input_dim: int = train.shape[1]
    n_learned_features = int(autoencoder_input_dim * int(args.expansion))
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                    n_learned_features=n_learned_features, n_components=1).cuda()
    print(f"Autoencoder created at {time() - start_time} seconds")
    print(f"------------Getting Image activations from model: {args.backbone}")

    # We use a loss reducer, which simply adds up the losses from the underlying loss functions.
    loss = LossReducer(LearnedActivationsL1Loss(
        l1_coefficient=float(args.l1), ), L2ReconstructionLoss(), )
    print(f"Loss created at {time() - start_time} seconds")

    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=float(5e-4),
        betas=(0.9,
               0.999),
        eps=1e-8,
        weight_decay=0.0,
        has_components_dim=True,
    )

    print(f"Optimizer created at {time() - start_time} seconds")
    actual_resample_interval = 1
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=math.inf,
        n_learned_features=n_learned_features, resample_epoch_freq=500000,
        resample_dataset_size=train.shape[0],
    )

    print(f"Activation resampler created at {time() - start_time} seconds")

    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(data_dir_activations["img"]),
        loss=loss,
        optimizer=optimizer,
        device="cuda",
        args={},
    )
    print(f"Pipeline created at {time() - start_time} seconds")

    fnames = os.listdir(data_dir_activations["img"])

    train_fnames = []
    train_val_fnames = []
    for fname in fnames:
        if fname.startswith(f"train_val"):
            train_val_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))
        elif fname.startswith(f"train"):
            train_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))

    print(f"Train and Train_val fnames created at {time() - start_time} seconds")

    # It takes the train activations and inside split it into train_activations and train_val_activations
    pipeline.run_pipeline(
        train_batch_size=int(4096),
        checkpoint_frequency=500000,
        val_frequency=50000,
        num_epochs=200,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=500000,
    )

    print(f"-------total time taken------ {np.round(time() - start_time, 3)}")

    index = []
    learned_activations = []
    for batch_idx in range(len(train) // 512 + 1):
        train_batch = train[batch_idx * 512: (batch_idx + 1) * 512].cuda()
        learned_activations_batch = autoencoder(train_batch).learned_activations[:, 0, :].detach().cpu()
        concept_batch = autoencoder(train_batch)
        index_batch = torch.abs(concept_batch.learned_activations).sum(dim=[0, 1]).cpu() > 0
        index.append(index_batch.unsqueeze(0))
        learned_activations.append(learned_activations_batch)
    learned_activations = torch.concat(learned_activations, dim=0)
    index = torch.concat(index, dim=0)
    index = index.sum(dim=0) > 0

    np.save(save_dir_activations + "/learned_activation.npy", learned_activations.detach().cpu().numpy())

    # ========== Save SAE Checkpoint ==========
    checkpoint_dir = save_dir_activations + "/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': autoencoder.state_dict(),
        'model_config': {
            'n_input_features': autoencoder_input_dim,
            'n_learned_features': n_learned_features,
            'n_components': 1
        },
        'train_shape': train.shape,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, 'autoencoder.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")