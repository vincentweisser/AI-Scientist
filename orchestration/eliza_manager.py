import json
import os
import os.path as osp
import shutil
from datetime import datetime

class ElizaManager:
    """Minimal experiment orchestration manager"""
    
    def __init__(self, base_dir, results_dir):
        self.base_dir = base_dir
        self.results_dir = results_dir
        
    def run_experiment(self, config):
        """Run a single NanoGPT experiment with the given configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_nanogpt_experiment"
        exp_dir = osp.join(self.results_dir, exp_name)
        
        # Copy experiment files
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copytree(self.base_dir, exp_dir, dirs_exist_ok=True)
        
        # Save experiment config
        config_path = osp.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        # Run experiment
        cmd = f"python experiment.py --out_dir {exp_dir} --config {config_path}"
        os.system(cmd)
        
        # Load and return results
        results_path = osp.join(exp_dir, "final_info.json")
        if osp.exists(results_path):
            with open(results_path, "r") as f:
                return json.load(f)
        return None
        
    def get_best_result(self):
        """Get the best result across all experiments"""
        best_loss = float("inf")
        best_result = None
        
        for exp_dir in os.listdir(self.results_dir):
            results_path = osp.join(self.results_dir, exp_dir, "final_info.json")
            if osp.exists(results_path):
                with open(results_path, "r") as f:
                    result = json.load(f)
                    if result["best_val_loss"] < best_loss:
                        best_loss = result["best_val_loss"]
                        best_result = result
        
        return best_result
