import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_dropout(mat: torch.Tensor, p: float, training: bool):
    """Dropout for sparse COO tensor."""
    if (not training) or p <= 0.0:
        return mat
    mat = mat.coalesce()
    idx = mat.indices()
    val = F.dropout(mat.values(), p=p, training=True)
    return torch.sparse_coo_tensor(idx, val, mat.size(), device=mat.device).coalesce()

class CoPLGCF_GCN(nn.Module):
    """
    Standard GCN implementation for CoPL.
    
    Difference from original CoPLGCF:
    - Instead of separate Weights (W_pos, W_neg, W_self...),
    - It uses a SINGLE shared Weight matrix per layer (Standard GCN).
    - It treats the graph as one unified signed graph: 
      A_total = [[0, A_pos - A_neg], [A_pos.T - A_neg.T, lambda * A_ii]]
    """

    def __init__(
        self,
        n_u: int,
        n_i: int,
        d: int,
        pos_adj_norm: torch.Tensor,
        neg_adj_norm: torch.Tensor,
        dropout: float = 0.1,
        l: int = 3,
        item_item_adj_norm: torch.Tensor | None = None,
        item_item_weight: float = 1.0,
    ):
        super().__init__()
        self.n_u, self.n_i, self.d = n_u, n_i, d
        self.l = l
        self.dropout = float(dropout)
        self.item_item_weight = float(item_item_weight)

        # Buffers for adjacencies
        self.register_buffer("pos_adj_norm", pos_adj_norm.coalesce().float())
        self.register_buffer("neg_adj_norm", neg_adj_norm.coalesce().float())
        
        if item_item_adj_norm is not None:
            self.register_buffer("item_item_adj_norm", item_item_adj_norm.coalesce().float())
        else:
            self.item_item_adj_norm = None

        self.act = nn.LeakyReLU(0.2)

        # Initial Embeddings (Learnable features)
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # GCN Weights: ONE Matrix per layer (Shared across all edge types)
        # H(l+1) = sigma( A_hat * H(l) * W(l) )
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])

    def encode_graph(self, test: bool = False):
        """
        Simulates Unified GCN propagation:
        H_new = Activation( (A_pos - A_neg + A_ii) @ H @ W )
        """
        E_u = self.E_u_0
        E_i = self.E_i_0

        for layer_idx, linear in enumerate(self.layers):
            # 1. Apply Dropout to Adjacencies
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)
            
            # 2. Aggregation (Simulating Unified Matrix Multiplication)
            # -------------------------------------------------------
            # User Node Aggregation:
            # neighbors are items. A_total_row_u = [0, A_pos - A_neg]
            # msg_u = (A_pos * E_i) - (A_neg * E_i)
            # -------------------------------------------------------
            msg_u = torch.spmm(Apos, E_i) - torch.spmm(Aneg, E_i)

            # -------------------------------------------------------
            # Item Node Aggregation:
            # neighbors are users AND items. A_total_row_i = [A_pos.T - A_neg.T, lambda * A_ii]
            # msg_i = (A_pos.T * E_u) - (A_neg.T * E_u) + (lambda * A_ii * E_i)
            # -------------------------------------------------------
            msg_i = torch.spmm(Apos.transpose(0, 1), E_u) - torch.spmm(Aneg.transpose(0, 1), E_u)
            
            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                msg_i = msg_i + self.item_item_weight * torch.spmm(Aii, E_i)

            # 3. Transformation (Linear) & Activation
            # GCN: H_new = Act( Msg @ W )
            # Note: usually GCN adds self-loops (H_self). 
            # If adj_norm doesn't have self-loops, we can add E_u explicitly (Residual) or skip.
            # Here we follow strict GCN: transformation is applied to the aggregated message.
            # To prevent vanishing gradients if node degree is 0, we typically add self-loop logic.
            # We will add E_prev (Residual connection) which is common in deep GCNs (ResGCN).
            
            E_u_new = linear(msg_u + E_u) # Adding E_u effectively acts as self-loop
            E_i_new = linear(msg_i + E_i) 

            E_u = self.act(E_u_new)
            E_i = self.act(E_i_new)

        return F.normalize(E_u, dim=-1), E_i

    @staticmethod
    def weighted_bce_with_logits(logits, labels, pos_weight=None, sample_weight=None):
        labels = labels.float()
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="none"
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        if sample_weight is not None:
            loss = loss * sample_weight

        return loss.mean()

    def forward_pointwise(
        self,
        uids: torch.Tensor,
        iids: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        test: bool = False,
        lambda_reg: float = 1e-6,
    ):
        E_u, E_i = self.encode_graph(test=test)

        u_emb = E_u[uids]
        i_emb = E_i[iids]
        logits = (u_emb * i_emb).sum(-1)

        loss_bce = self.weighted_bce_with_logits(
            logits, labels, pos_weight=pos_weight, sample_weight=sample_weight
        )
        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        loss = loss_bce + lambda_reg * loss_reg

        return loss, logits