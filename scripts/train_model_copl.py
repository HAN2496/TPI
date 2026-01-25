import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.model.CoPL.data import CoPLDataProcessor
from src.model.CoPL.gcf import CoPLGCF
from src.model.CoPL.rm import DriverRewardModel
from src.model.CoPL.visualization import visualize_user_embeddings, visualize_item_embeddings, plot_reward_distribution
from src.utils.visualization import plot_roc_curve

# --- Configuration ---
FLAGS = {
    # Data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_names": ["강신길"],
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "time_range": (5, 7),
    "downsample": 1,
    "device": "cuda",
    
    # GCF Settings
    "gcf_dim": 512,
    "gcf_layers": 3,
    "gcf_lr": 0.0001,
    "gcf_epochs": 100,
    "gcf_batch_size": 2048,

    # RM Settings
    "rm_hidden_dim": 64,
    "rm_lr": 0.001,
    "rm_epochs": 200,
    "rm_batch_size": 64,
    
    # Test Adaptation Settings
    "query_ratio": 0.3,
    "adapt_lr": 0.01,
    "adapt_epochs": 200,
}

def train_gcf(model, adj_mat, interactions, num_users, num_items, device):
    """
    Trains the Graph Collaborative Filtering (GCF) model.
    """
    optimizer = optim.Adam(model.parameters(), lr=FLAGS['gcf_lr'])
    model.train()
    
    # Create a set for fast lookup of positive (u, i) pairs
    pos_set = set([(u, i) for u, i in interactions])
    
    users = np.array([u for u, i in interactions])
    items = np.array([i for u, i in interactions])
    
    n_interactions = len(interactions)
    
    print(f"\n[GCF] Starting Training on {device}...")
    
    for epoch in range(FLAGS['gcf_epochs']):
        # Shuffle interactions
        perm = np.random.permutation(n_interactions)
        users = users[perm]
        items = items[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_interactions, FLAGS['gcf_batch_size']):
            batch_users = users[i:i+FLAGS['gcf_batch_size']]
            batch_pos_items = items[i:i+FLAGS['gcf_batch_size']]
            
            # Sample negatives (Uniform sampling)
            batch_neg_items = []
            for u in batch_users:
                while True:
                    neg_i = np.random.randint(0, num_items)
                    # Rejection sampling: ensure it's not a positive edge
                    if (u, neg_i) not in pos_set:
                        batch_neg_items.append(neg_i)
                        break
            
            batch_users_t = torch.LongTensor(batch_users).to(device)
            batch_pos_t = torch.LongTensor(batch_pos_items).to(device)
            batch_neg_t = torch.LongTensor(batch_neg_items).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through Graph (LightGCN)
            # Returns all embeddings
            u_emb_all, i_emb_all = model(adj_mat)
            
            # Compute BPR Loss
            loss, reg_loss = model.get_loss(u_emb_all, i_emb_all, batch_users_t, batch_pos_t, batch_neg_t)
            
            total = loss + 1e-4 * reg_loss
            
            total.backward()
            optimizer.step()
            
            total_loss += total.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches
        if (epoch+1) % 10 == 0:
            print(f"  GCF Epoch {epoch+1}/{FLAGS['gcf_epochs']} Loss: {avg_loss:.4f}")

    # Return final learned user embeddings
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(adj_mat)
    return u_emb, i_emb

def train_rm(model, dataset, user_embeddings, device):
    """
    Trains the Reward Model (RM) using the fixed user embeddings from GCF.
    """
    optimizer = optim.Adam(model.parameters(), lr=FLAGS['rm_lr'])
    loader = DataLoader(dataset, batch_size=FLAGS['rm_batch_size'], shuffle=True)
    
    # Binary Cross Entropy with Logits (for stability)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    print(f"\n[RM] Starting Reward Model Training on {device}...")
    
    # Detach user embeddings (treat as fixed inputs features)
    user_embeddings = user_embeddings.detach()
    
    for epoch in range(FLAGS['rm_epochs']):
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            x = batch['feature'].to(device)     # (B, T, D)
            u_ids = batch['user_id'].to(device) # (B, 1)
            y = batch['label'].to(device).unsqueeze(1) # (B, 1)
            
            # Look up embeddings for the batch's users
            u_embs = user_embeddings[u_ids]
            
            optimizer.zero_grad()
            
            # Predict Reward
            logits = model(x, u_embs)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches
        if (epoch+1) % 10 == 0:
            print(f"  RM Epoch {epoch+1}/{FLAGS['rm_epochs']} Loss: {avg_loss:.4f}")

def adapt_and_evaluate(rm_model, test_X, test_y, device, driver_name):
    """
    Adapts to a new user by optimizing a new user embedding on a query set,
    then evaluating on the rest.
    """
    print(f"\n[Adaptation] Processing {driver_name}...")
    
    # Split into Query (Adaptation) and Target (Evaluation)
    X_query, X_target, y_query, y_target = train_test_split(
        test_X, test_y, train_size=FLAGS['query_ratio'], random_state=42, stratify=test_y
    )
    
    # Convert to Tensors
    X_query_t = torch.FloatTensor(X_query).to(device)
    y_query_t = torch.FloatTensor(y_query).unsqueeze(1).to(device)
    X_target_t = torch.FloatTensor(X_target).to(device)
    # y_target stays numpy for metric calculation
    
    # Initialize new user embedding (Randomly)
    # Shape: (1, Embedding_Dim)
    user_emb_new = torch.nn.Parameter(torch.randn(1, FLAGS['gcf_dim']).to(device))
    
    # Optimize ONLY the user embedding
    optimizer = optim.Adam([user_emb_new], lr=FLAGS['adapt_lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    rm_model.eval() # Freeze RM layers (batchnorm/dropout behavior)
    # Note: We don't set requires_grad=False for RM params because optimizer only has user_emb_new
    
    print(f"  Optimizing user embedding for {FLAGS['adapt_epochs']} epochs...")
    for epoch in range(FLAGS['adapt_epochs']):
        optimizer.zero_grad()
        
        # Expand user_emb to batch size
        u_emb_batch = user_emb_new.expand(X_query_t.size(0), -1)
        
        logits = rm_model(X_query_t, u_emb_batch)
        loss = criterion(logits, y_query_t)
        
        loss.backward()
        optimizer.step()
        
    # Evaluate on Target Set
    with torch.no_grad():
        u_emb_eval = user_emb_new.expand(X_target_t.size(0), -1)
        logits_test = rm_model(X_target_t, u_emb_eval)
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        
    auroc = roc_auc_score(y_target, probs_test)
    print(f"  Result for {driver_name}: AUROC = {auroc:.4f}")
    
    # Visualize
    plot_roc_curve(y_target, probs_test, 
                   save_path=f"artifacts/copl/roc_{driver_name}.png", 
                   title=f"ROC - {driver_name} (CoPL Adaptation)")
    
    plot_reward_distribution(logits_test.cpu().numpy(), y_target, driver_name,
                             save_path=f"artifacts/copl/rewards_{driver_name}.png")

def main():
    os.makedirs("artifacts/copl", exist_ok=True)
    device = torch.device(FLAGS['device'])
    print(f"Device: {device}")
    
    # 1. Data Processing (Training)
    processor = CoPLDataProcessor(
        feature_list=FLAGS['features'],
        time_range=FLAGS['time_range'],
        downsample=FLAGS['downsample']
    )
    dataset = processor.load_data(FLAGS['train_driver_names'])
    
    # Get Graph for GCF
    adj_mat = processor.get_adj_mat()
    if adj_mat is not None:
        adj_mat = adj_mat.to(device)
    else:
        print("Error: Empty adjacency matrix.")
        return

    # 2. GCF Training
    print(f"Initializing GCF: Users={processor.num_users}, Items={processor.num_items}")
    gcf_model = CoPLGCF(
        num_users=processor.num_users, 
        num_items=processor.num_items,
        embedding_dim=FLAGS['gcf_dim'],
        n_layers=FLAGS['gcf_layers']
    ).to(device)
    
    learned_user_embs, learned_item_embs = train_gcf(
        gcf_model, adj_mat, processor.interactions, 
        processor.num_users, processor.num_items, device
    )
    
    # Visualize Learned User Embeddings
    visualize_user_embeddings(learned_user_embs[:processor.num_users], processor.user_map, 
                              save_path="artifacts/copl/user_embeddings.png")

    # Visualize Learned Item Embeddings (Episodes)
    # We use processor.train_user_ids because items are added sequentially aligned with this list
    visualize_item_embeddings(learned_item_embs[:processor.num_items], processor.train_user_ids, processor.user_map,
                              save_path="artifacts/copl/item_embeddings.png")
    
    # 3. Reward Model Training
    feature_dim = dataset[0]['feature'].shape[1] 
    
    print(f"Initializing RM: Input Feature Dim={feature_dim}, User Emb Dim={FLAGS['gcf_dim']}")
    rm_model = DriverRewardModel(
        input_dim=feature_dim,
        user_emb_dim=FLAGS['gcf_dim'],
        hidden_dim=FLAGS['rm_hidden_dim']
    ).to(device)
    
    train_rm(rm_model, dataset, learned_user_embs, device)
    
    # Save Trained Models
    torch.save(gcf_model.state_dict(), "artifacts/copl/gcf_model.pt")
    torch.save(rm_model.state_dict(), "artifacts/copl/rm_model.pt")
    
    # 4. Evaluation on Test Drivers (Unseen Adaptation)
    print("\n--- Starting Evaluation on Test Drivers ---")
    
    for test_driver in FLAGS['test_driver_names']:
        X_test, y_test = processor.load_test_data(test_driver)
        if len(X_test) == 0:
            print(f"No data for {test_driver}, skipping.")
            continue
            
        adapt_and_evaluate(rm_model, X_test, y_test, device, test_driver)
    
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()