import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from src.utils.utils import _load_dataset_sequences

class CoPLDataset(Dataset):
    def __init__(self, features, user_ids, labels):
        self.features = torch.FloatTensor(np.array(features))
        self.user_ids = torch.LongTensor(np.array(user_ids))
        self.labels = torch.FloatTensor(np.array(labels))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'feature': self.features[idx],
            'user_id': self.user_ids[idx],
            'label': self.labels[idx]
        }

class CoPLDataProcessor:
    def __init__(self, feature_list, time_range=(5, 7), downsample=1):
        self.feature_list = feature_list
        self.time_range = time_range
        self.downsample = downsample
        
        self.user_map = {} # driver_name -> user_id
        
        # Lists to store training data
        self.train_features = []
        self.train_user_ids = []
        self.train_labels = []
        
        self.num_users = 0
        self.num_items = 0
        
        self.interactions = [] # List of (user_id, item_id)
        
        self.norm_mean = None
        self.norm_std = None

    def load_data(self, driver_names, normalize=True):
        print("Loading data for CoPL...")
        raw_data = {}
        all_obs = []
        config = {'features': self.feature_list}
        
        # 1. Load Raw Data
        for driver in driver_names:
            if driver not in self.user_map:
                self.user_map[driver] = len(self.user_map)
            
            X, y = _load_dataset_sequences(driver, self.time_range, self.downsample, config)
            raw_data[driver] = (X, y)
            all_obs.append(X)
            
        # 2. Compute Normalization
        if normalize:
            concat_obs = np.concatenate(all_obs, axis=0)
            self.norm_mean = np.mean(concat_obs, axis=(0, 1))
            self.norm_std = np.std(concat_obs, axis=(0, 1)) + 1e-6
            print(f"Normalization Mean: {self.norm_mean}, Std: {self.norm_std}")
        
        # 3. Process Episodes into Items and Graph
        self.num_users = len(self.user_map)
        # We start item IDs after the last user ID if we were doing a single graph, 
        # but usually for implementation convenience, user IDs are 0..Nu-1 and Item IDs are 0..Ni-1.
        # The Adjacency Matrix construction handles the offsetting.
        
        current_item_id = 0
        
        for driver in driver_names:
            X, y = raw_data[driver]
            if normalize:
                X = (X - self.norm_mean) / self.norm_std
                
            u_id = self.user_map[driver]
            
            for i in range(len(X)):
                # Each episode is a unique item
                # In this specific formulation, we treat every training episode as a distinct item node.
                
                # Store for RM Training
                self.train_features.append(X[i])
                self.train_user_ids.append(u_id)
                self.train_labels.append(y[i])
                
                # If Positive, add edge to Graph
                if y[i] == 1:
                    self.interactions.append((u_id, current_item_id))
                
                current_item_id += 1
        
        self.num_items = current_item_id
        
        print(f"CoPL Data Processed: {self.num_users} Users, {self.num_items} Items")
        print(f"Graph Edges (Positive Interactions): {len(self.interactions)}")
        
        return self.get_dataset()

    def load_test_data(self, driver_name):
        """
        Loads data for a test driver.
        Returns:
            X: features (normalized if training data was normalized)
            y: labels
        """
        print(f"Loading test data for {driver_name}...")
        config = {'features': self.feature_list}
        
        X, y = _load_dataset_sequences(driver_name, self.time_range, self.downsample, config)
        
        if self.norm_mean is not None and self.norm_std is not None:
            X = (X - self.norm_mean) / self.norm_std
            
        return X, y

    def get_dataset(self):
        return CoPLDataset(self.train_features, self.train_user_ids, self.train_labels)
        
    def get_adj_mat(self):
        """
        Constructs the Normalized Adjacency Matrix for LightGCN.
        Matrix shape: (N_users + N_items, N_users + N_items)
        """
        if self.num_users == 0 or self.num_items == 0:
            return None
            
        print("Constructing Adjacency Matrix...")
        
        # User nodes: 0 to N_users-1
        # Item nodes: N_users to N_users + N_items - 1
        
        user_nodes = [u for u, i in self.interactions]
        item_nodes = [i + self.num_users for u, i in self.interactions]
        
        # Create symmetric adjacency matrix
        row_idx = user_nodes + item_nodes
        col_idx = item_nodes + user_nodes
        data = np.ones(len(row_idx))
        
        n_nodes = self.num_users + self.num_items
        adj_mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_nodes, n_nodes))
        
        # Normalization: D^-0.5 * A * D^-0.5
        rowsum = np.array(adj_mat.sum(1))
        # Add epsilon to avoid divide by zero for isolated nodes
        d_inv_sqrt = np.power(np.maximum(rowsum, 1e-12), -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        # Convert to PyTorch Sparse Tensor
        coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))
