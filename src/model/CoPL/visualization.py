import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from src.utils.visualization import plot_roc_curve

def visualize_user_embeddings(user_embeddings, user_map, save_path=None):
    """
    Visualizes user embeddings using t-SNE.
    Args:
        user_embeddings: (num_users, embedding_dim) torch tensor
        user_map: dict mapping driver_name -> user_id
        save_path: path to save the plot
    """
    # Convert to numpy
    embeddings_np = user_embeddings.detach().cpu().numpy()
    
    # t-SNE reduction
    n_samples = embeddings_np.shape[0]
    perplexity = min(30, n_samples - 1) # Perplexity must be less than n_samples
    
    if n_samples < 2:
        print("Not enough samples for t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Reverse map for labels
    id_to_user = {v: k for k, v in user_map.items()}
    
    for i in range(n_samples):
        x, y = embeddings_2d[i]
        label = id_to_user.get(i, f"User {i}")
        plt.scatter(x, y, alpha=0.7)
        plt.text(x, y, label, fontsize=9)
        
    plt.title("User Embeddings t-SNE")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_item_embeddings(item_embeddings, item_labels, user_map, save_path=None):
    """
    Visualizes item (episode) embeddings using t-SNE, colored by User ID.
    Args:
        item_embeddings: (num_items, embedding_dim) torch tensor
        item_labels: list or array of user_ids corresponding to each item
        user_map: dict mapping driver_name -> user_id
        save_path: path to save the plot
    """
    # Convert to numpy
    embeddings_np = item_embeddings.detach().cpu().numpy()
    labels_np = np.array(item_labels)
    
    # t-SNE reduction
    n_samples = embeddings_np.shape[0]
    perplexity = min(30, n_samples - 1)
    
    if n_samples < 2:
        print("Not enough samples for Item t-SNE.")
        return

    print(f"Running t-SNE on {n_samples} items...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Reverse map for legend
    id_to_user = {v: k for k, v in user_map.items()}
    unique_users = np.unique(labels_np)
    
    # Color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_users)))
    
    for i, u_id in enumerate(unique_users):
        indices = labels_np == u_id
        user_name = id_to_user.get(u_id, f"User {u_id}")
        
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                    alpha=0.6, label=user_name, s=10, color=colors[i])
        
    plt.title("Item (Episode) Embeddings t-SNE by Driver")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_reward_distribution(rewards, labels, driver_name, save_path=None):
    """
    Plots histogram of rewards for positive (Good) vs negative (Bad) labels.
    """
    rewards = np.array(rewards).flatten()
    labels = np.array(labels).flatten()
    
    pos_rewards = rewards[labels == 1]
    neg_rewards = rewards[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_rewards, bins=30, alpha=0.5, label='Good', color='red', density=True)
    plt.hist(neg_rewards, bins=30, alpha=0.5, label='Bad', color='blue', density=True)
    
    plt.title(f"Reward Distribution - {driver_name}")
    plt.xlabel("Predicted Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
