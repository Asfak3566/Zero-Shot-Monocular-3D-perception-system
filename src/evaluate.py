import os
import yaml
import torch
from torch.utils.data import DataLoader
from dataset_multi_agent import MultiAgentDataset
from model_multi_agent_sota import MultiAgentTransformerSOTA
from tqdm import tqdm
import numpy as np

def evaluate():
    config_path = "experiments/model_d_multi_agent_sota/config_model_d_multi_agent.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    waymo_path = "twoDthreeDnet/dataset_generation/from_waymo"
    dataset = MultiAgentDataset(waymo_path, config, split='val')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MultiAgentTransformerSOTA(config).to(device)
    checkpoint_path = "experiments/model_d_multi_agent_sota/checkpoints/best_model_d.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: No checkpoint found. Evaluating random model.")
        
    model.eval()
    
    all_mae = []
    all_ori_err = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            agent_mask = batch['agent_mask'].to(device)
            visibility_mask = X[..., 17]
            
            pred = model(X, agent_mask, visibility_mask=visibility_mask)
            
            # MAE (x, y) - scale back by 100
            diff = (pred[..., :2] - Y[..., :2]) * 100.0
            mae = torch.norm(diff, dim=-1) # (B, N, P)
            
            # Orientation error
            # cos(theta_err) = cos(p)*cos(t) + sin(p)*sin(t)
            cos_sim = (pred[..., 6:8] * Y[..., 6:8]).sum(dim=-1).clamp(-1, 1)
            ori_err = torch.acos(cos_sim) * 180.0 / np.pi
            
            # Apply masks
            for b in range(X.shape[0]):
                for n in range(X.shape[1]):
                    if agent_mask[b, n] > 0.5:
                        all_mae.append(mae[b, n].cpu().numpy())
                        all_ori_err.append(ori_err[b, n].cpu().numpy())
    
    all_mae = np.array(all_mae)
    all_ori_err = np.array(all_ori_err)
    
    results = {
        'MAE_mean': np.mean(all_mae),
        'MAE_median': np.median(all_mae),
        'MAE_std': np.std(all_mae),
        'OriErr_mean': np.mean(all_ori_err),
        'OriErr_median': np.median(all_ori_err)
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    os.makedirs("experiments/model_d_multi_agent_sota/results", exist_ok=True)
    with open("experiments/model_d_multi_agent_sota/results/eval_metrics.yaml", 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    evaluate()
