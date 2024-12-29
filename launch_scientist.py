"""Streamlined launcher for NanoGPT experiments"""
# Set environment variables for optimized CUDA memory allocation and debugging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import argparse
import os.path as osp

from ai_scientist.orchestration.eliza_manager import ElizaManager
from ai_scientist.config.nanogpt_config import load_config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run NanoGPT experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config/nanogpt_config.json",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to store experiment results",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up experiment manager
    base_dir = osp.join(os.getcwd(), "templates/nanoGPT")
    results_dir = osp.join(os.getcwd(), args.out_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    manager = ElizaManager(base_dir, results_dir)
    
    # Run experiment
    print(f"Starting experiment with config: {config}")
    result = manager.run_experiment(config)
    
    if result:
        print(f"Experiment completed successfully!")
        print(f"Best validation loss: {result['best_val_loss']}")
        print(f"Total training time: {result['total_train_time']} seconds")
    else:
        print("Experiment failed!")


if __name__ == "__main__":
    main()
