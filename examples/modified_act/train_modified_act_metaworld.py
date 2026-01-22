#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for Modified ACT (without joint states) on MetaWorld reach-v3.

This script demonstrates how to train the modified ACT policy that:
- VAE Encoder: Takes (images, actions) instead of (joint_states, actions)
- Decoder: Conditions on (z, images) instead of (z, images, joint_states)

Usage:
    python train_modified_act_metaworld.py

Or using lerobot-train CLI (recommended):
    See the training commands at the bottom of this file.
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    """Train modified ACT policy on metaworld-reach-v3 dataset."""
    
    # Configuration
    output_directory = Path("outputs/train/modified_act_metaworld_reach")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training hyperparameters
    training_steps = 50000
    batch_size = 8
    log_freq = 100
    save_freq = 5000
    eval_freq = 5000
    
    # Dataset: aadarshram/metaworld-reach-v3
    # Features:
    #   - observation.image: [480, 480, 3]
    #   - observation.state: [4] (x, y, z, gripper) - NOT USED in modified ACT
    #   - observation.environment_state: [39]
    #   - action: [4]
    dataset_repo_id = "aadarshram/metaworld-reach-v3"
    
    print(f"Loading dataset metadata from {dataset_repo_id}...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    
    # Convert dataset features to policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # Separate input and output features
    # IMPORTANT: For modified ACT, we EXCLUDE observation.state from input features
    # The modified ACT only uses images (and optionally environment_state)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    
    # Filter input features - exclude observation.state for modified ACT
    input_features = {}
    for key, ft in features.items():
        if key in output_features:
            continue
        # Skip observation.state - modified ACT doesn't use joint states
        if key == "observation.state":
            print(f"  [SKIPPED] {key}: {ft.shape} (joint states not used in modified ACT)")
            continue
        input_features[key] = ft
        print(f"  [INPUT] {key}: {ft.shape}, type: {ft.type}")
    
    for key, ft in output_features.items():
        print(f"  [OUTPUT] {key}: {ft.shape}, type: {ft.type}")
    
    # Create ACT configuration
    # The modified ACT requires image inputs and doesn't use observation.state
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # ACT-specific parameters
        chunk_size=50,  # Action chunk size (predict 50 steps at a time)
        n_action_steps=50,  # Execute all predicted actions
        # VAE parameters
        use_vae=True,
        latent_dim=32,
        kl_weight=10.0,
        # Transformer architecture
        dim_model=256,  # Reduced for faster training on smaller dataset
        n_heads=8,
        dim_feedforward=1024,
        n_encoder_layers=4,
        n_decoder_layers=1,
        n_vae_encoder_layers=4,
        # Vision backbone
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        # Training
        dropout=0.1,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        optimizer_lr_backbone=1e-5,
        # Device
        device=str(device),
    )
    
    print("\nCreating Modified ACT Policy...")
    print("  - VAE Encoder inputs: (images, actions) - NO joint states")
    print("  - Decoder inputs: (z, images) - NO joint states")
    
    # Create policy
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    
    print(f"\nPolicy created with {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Create pre/post processors for normalization
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # Set up delta timestamps for ACT
    # ACT uses n_obs_steps=1, so we only need current frame
    delta_timestamps = {
        "observation.image": [0.0],  # Current frame only
        # Note: observation.state is NOT included - modified ACT doesn't use it
        "action": [i / dataset_metadata.fps for i in range(cfg.chunk_size)],
    }
    
    # Add environment_state if present in features
    if "observation.environment_state" in input_features:
        delta_timestamps["observation.environment_state"] = [0.0]
    
    print(f"\nDelta timestamps: {delta_timestamps}")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create optimizer (using policy's preset or custom)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {training_steps} steps")
    print(f"Batch size: {batch_size}")
    print(f"Log frequency: {log_freq}")
    print(f"Save frequency: {save_freq}")
    print(f"{'='*60}\n")
    
    step = 0
    running_loss = 0.0
    running_l1_loss = 0.0
    running_kl_loss = 0.0
    
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            
            # Preprocess batch (normalization, device placement)
            batch = preprocessor(batch)
            
            # Forward pass and compute loss
            loss, loss_dict = policy.forward(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Accumulate losses for logging
            running_loss += loss.item()
            running_l1_loss += loss_dict.get("l1_loss", 0.0)
            running_kl_loss += loss_dict.get("kld_loss", 0.0)
            
            step += 1
            
            # Log progress
            if step % log_freq == 0:
                avg_loss = running_loss / log_freq
                avg_l1 = running_l1_loss / log_freq
                avg_kl = running_kl_loss / log_freq
                
                print(f"Step {step}/{training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"L1: {avg_l1:.4f} | "
                      f"KL: {avg_kl:.4f}")
                
                running_loss = 0.0
                running_l1_loss = 0.0
                running_kl_loss = 0.0
            
            # Save checkpoint
            if step % save_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint_{step:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                policy.save_pretrained(checkpoint_dir)
                
                # Save optimizer state
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                
                print(f"  → Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_dir = output_directory / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    torch.save(optimizer.state_dict(), final_dir / "optimizer.pt")
    print(f"\n{'='*60}")
    print(f"Training complete! Final model saved to {final_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# =============================================================================
# ALTERNATIVE: Using lerobot-train CLI (Recommended for production)
# =============================================================================
#
# TRAINING COMMAND:
# -----------------
# lerobot-train \
#     --policy.type=act \
#     --policy.chunk_size=50 \
#     --policy.n_action_steps=50 \
#     --policy.use_vae=true \
#     --policy.latent_dim=32 \
#     --policy.kl_weight=10.0 \
#     --policy.dim_model=256 \
#     --policy.vision_backbone=resnet18 \
#     --policy.device=cuda \
#     --dataset.repo_id=aadarshram/metaworld-reach-v3 \
#     --env.type=metaworld \
#     --env.task=reach-v3 \
#     --output_dir=outputs/train/modified_act_metaworld_reach \
#     --steps=50000 \
#     --batch_size=8 \
#     --eval_freq=5000 \
#     --save_freq=5000 \
#     --log_freq=100 \
#     --eval.batch_size=1 \
#     --eval.n_episodes=10
#
# NOTES:
# - The modified ACT will automatically skip observation.state since images are provided
# - Make sure MetaWorld is properly installed: pip install metaworld
# - The --env.type=metaworld enables evaluation during training
#
# =============================================================================
# EVALUATION COMMAND (after training):
# =============================================================================
#
# lerobot-eval \
#     --policy.path=outputs/train/modified_act_metaworld_reach/final_model \
#     --env.type=metaworld \
#     --env.task=reach-v3 \
#     --eval.batch_size=1 \
#     --eval.n_episodes=50 \
#     --policy.device=cuda
#
# This will report:
# - Success rate
# - Average return  
# - Episode length
# =============================================================================
