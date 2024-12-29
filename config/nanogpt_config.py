"""Configuration loader for NanoGPT experiments"""
import json
import os.path as osp

def load_config(config_path):
    """Load NanoGPT configuration from JSON file"""
    if not osp.exists(config_path):
        return {
            "n_layer": 12,
            "n_head": 6,
            "n_embd": 768,
            "block_size": 128,
            "batch_size": 8,
            "learning_rate": 5e-4,
            "max_iters": 2000,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "decay_lr": True,
            "warmup_iters": 100,
            "lr_decay_iters": 2000,
            "min_lr": 1e-5,
            "dropout": 0.0,
            "compile": True,
            "rope_max_seq_len": 65536,
            "remove_8th_attention": True,
            "use_muon_optimizer": True,
            "value_embedding_pattern": [0, 1, 2, None, None, None, None, None, None, 0, 1, 2]
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)
