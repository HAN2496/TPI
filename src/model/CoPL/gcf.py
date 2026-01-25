import torch
import torch.nn as nn
import torch.nn.functional as F

class CoPLGCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(CoPLGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initial Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, adj_mat):
        """
        Propagate embeddings through the graph.
        Returns:
            users_emb: (num_users, embedding_dim)
            items_emb: (num_items, embedding_dim)
        """
        # Initial embeddings
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        all_embeddings = [ego_embeddings]
        
        for k in range(self.n_layers):
            # Graph Convolution: A * E
            if adj_mat is not None:
                ego_embeddings = torch.sparse.mm(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # LightGCN aggregation: Mean of layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        users_emb, items_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        
        return users_emb, items_emb

    def get_loss(self, users_emb, items_emb, users_idx, pos_items_idx, neg_items_idx):
        """
        Bayesian Personalized Ranking (BPR) Loss
        """
        u_emb = users_emb[users_idx]
        pos_i_emb = items_emb[pos_items_idx]
        neg_i_emb = items_emb[neg_items_idx]
        
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)
        
        # Softplus is a smooth approximation of ReLU, but here we use LogSigmoid for BPR
        # Loss = -ln(sigmoid(pos - neg))
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 Regularization
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) + 
                          pos_i_emb.norm(2).pow(2) + 
                          neg_i_emb.norm(2).pow(2)) / float(len(users_idx))
                          
        return loss, reg_loss
