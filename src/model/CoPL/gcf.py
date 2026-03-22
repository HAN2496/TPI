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


class CoPLGCF(nn.Module):
    """
    CoPL-GCF adapted for time-series items + binary (good/bad) labels.

    - user-item edges are split into positive/negative: A_pos, A_neg
    - item-item similarity edges: A_ii (precomputed, normalized)  <-- 핵심(3)
    - Training loss: pointwise BCE (no pairwise), supports imbalance via pos_weight / sample_weight
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

        self.pos_adj_norm = pos_adj_norm.coalesce().float()
        self.neg_adj_norm = neg_adj_norm.coalesce().float()
        self.item_item_adj_norm = None if item_item_adj_norm is None else item_item_adj_norm.coalesce().float()
        self.item_item_weight = float(item_item_weight)

        self.act = nn.LeakyReLU(0.2)

        # embeddings
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # Eq.(2) 스타일: typed message + hadamard message + self
        self.W_u_self = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_u_pos_1 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_u_pos_2 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_u_neg_3 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_u_neg_4 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])

        self.W_i_self = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_i_pos_1 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_i_pos_2 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_i_neg_3 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_i_neg_4 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])

        # item-item(sim) message (optional)
        self.W_i_ii_1 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])
        self.W_i_ii_2 = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])

        # cache
        self.E_u = None
        self.E_i = None

    def encode_graph(self, test: bool = False):
        """
        Run l-layer message passing to produce (E_u, E_i).
        test=True: no sparse dropout
        """
        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)

            Z_u_pos = torch.spmm(Apos, E_i_prev)                 # (n_u, d)
            Z_u_neg = torch.spmm(Aneg, E_i_prev)                 # (n_u, d)
            Z_i_pos = torch.spmm(Apos.transpose(0, 1), E_u_prev)  # (n_i, d)
            Z_i_neg = torch.spmm(Aneg.transpose(0, 1), E_u_prev)  # (n_i, d)

            m_u = (
                self.W_u_self[layer](E_u_prev)
                + self.W_u_pos_1[layer](Z_u_pos)
                + self.W_u_pos_2[layer](Z_u_pos * E_u_prev)
                + self.W_u_neg_3[layer](Z_u_neg)
                + self.W_u_neg_4[layer](Z_u_neg * E_u_prev)
            )

            m_i = (
                self.W_i_self[layer](E_i_prev)
                + self.W_i_pos_1[layer](Z_i_pos)
                + self.W_i_pos_2[layer](Z_i_pos * E_i_prev)
                + self.W_i_neg_3[layer](Z_i_neg)
                + self.W_i_neg_4[layer](Z_i_neg * E_i_prev)
            )

            # 핵심: item-item 메시지 추가
            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                Z_i_ii = torch.spmm(Aii, E_i_prev)  # (n_i, d)
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii) + self.W_i_ii_2[layer](Z_i_ii * E_i_prev)
                )

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        self.E_u = F.normalize(E_u_prev, dim=-1)
        self.E_i = E_i_prev
        return self.E_u, self.E_i

    @staticmethod
    def weighted_bce_with_logits(logits, labels, pos_weight=None, sample_weight=None):
        """
        logits: (B,)
        labels: (B,) in {0,1}
        pos_weight: scalar tensor (neg/pos) for BCEWithLogits
        sample_weight: (B,) optional per-example weight
        """
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
        """
        Pointwise training (no pairwise):
          p_hat = sigmoid(<e_u, e_i>)
          loss = weighted BCE + reg
        """
        E_u, E_i = self.encode_graph(test=test)

        u_emb = E_u[uids]
        i_emb = E_i[iids]
        logits = (u_emb * i_emb).sum(-1)  # (B,)

        loss_bce = self.weighted_bce_with_logits(
            logits, labels, pos_weight=pos_weight, sample_weight=sample_weight
        )
        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        loss = loss_bce + lambda_reg * loss_reg

        return loss, logits


class CoPLGCFCosine(CoPLGCF):
    """
    CoPL-GCF adapted for Cosine Embedding Loss.
    
    Changes:
    - User/Item embeddings are L2 normalized in encode_graph.
    - Loss function: CosineEmbeddingLoss (margin-based).
    - Prediction: Cosine Similarity (-1 to 1).
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
        margin: float = 0.5,
    ):
        super().__init__(
            n_u, n_i, d, pos_adj_norm, neg_adj_norm, dropout, l, item_item_adj_norm, item_item_weight
        )
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="none")

    def encode_graph(self, test: bool = False):
        """
        Run l-layer message passing to produce (E_u, E_i).
        Both are L2 normalized for Cosine Similarity.
        """
        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)

            Z_u_pos = torch.spmm(Apos, E_i_prev)
            Z_u_neg = torch.spmm(Aneg, E_i_prev)
            Z_i_pos = torch.spmm(Apos.transpose(0, 1), E_u_prev)
            Z_i_neg = torch.spmm(Aneg.transpose(0, 1), E_u_prev)

            m_u = (
                self.W_u_self[layer](E_u_prev)
                + self.W_u_pos_1[layer](Z_u_pos)
                + self.W_u_pos_2[layer](Z_u_pos * E_u_prev)
                + self.W_u_neg_3[layer](Z_u_neg)
                + self.W_u_neg_4[layer](Z_u_neg * E_u_prev)
            )

            m_i = (
                self.W_i_self[layer](E_i_prev)
                + self.W_i_pos_1[layer](Z_i_pos)
                + self.W_i_pos_2[layer](Z_i_pos * E_i_prev)
                + self.W_i_neg_3[layer](Z_i_neg)
                + self.W_i_neg_4[layer](Z_i_neg * E_i_prev)
            )

            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                Z_i_ii = torch.spmm(Aii, E_i_prev)
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii) + self.W_i_ii_2[layer](Z_i_ii * E_i_prev)
                )

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        # Normalize BOTH
        self.E_u = F.normalize(E_u_prev, dim=-1)
        self.E_i = F.normalize(E_i_prev, dim=-1)
        return self.E_u, self.E_i

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
        """
        Pointwise training using Cosine Embedding Loss.
        Target: 1 (Pos) or -1 (Neg).
        """
        E_u, E_i = self.encode_graph(test=test)

        u_emb = E_u[uids]
        i_emb = E_i[iids]
        
        # Cosine Similarity (-1 ~ 1)
        scores = (u_emb * i_emb).sum(-1)

        # Convert labels {0, 1} -> {-1, 1}
        target = labels.float()
        target[target == 0] = -1

        loss = self.loss_fn(u_emb, i_emb, target)

        if sample_weight is not None:
            loss = loss * sample_weight

        return loss.mean(), scores


class CoPLGCFPointwiseBPR(CoPLGCF):
    """
    CoPL-GCF adapted for Pointwise BPR Loss (Absolute BPR).
    
    Concept:
    - Positive score should be > 0.
    - Negative score should be < 0.
    - Loss: -ln(sigmoid(score)) if Pos, -ln(1-sigmoid(score)) if Neg.
    - Mathematically similar to BCE but emphasizes the "ranking" aspect against 0.
    """

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
        logits = (u_emb * i_emb).sum(-1)  # (B,)

        # Pointwise BPR Loss
        # Pos: maximize log(sigmoid(x))
        # Neg: maximize log(sigmoid(-x)) = log(1 - sigmoid(x))
        
        log_prob_pos = F.logsigmoid(logits)
        log_prob_neg = F.logsigmoid(-logits)
        
        # Select relevant term based on label
        loss = -(labels * log_prob_pos + (1 - labels) * log_prob_neg)

        if pos_weight is not None:
            # Apply pos_weight only to positive terms
            loss = -(labels * log_prob_pos * pos_weight + (1 - labels) * log_prob_neg)
        
        if sample_weight is not None:
            loss = loss * sample_weight

        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        loss = loss.mean() + lambda_reg * loss_reg

        return loss, logits


class CoPLGCFSoftmax(CoPLGCF):
    """
    CoPL-GCF adapted for In-batch Softmax Loss (Listwise Approximation).
    
    Concept:
    - Treats other items in the batch as negatives for the current user.
    - Effective for "Positive push, Negative pull" with implicit negatives.
    - Requires large batch size for best performance.
    """
    def __init__(self, *args, temperature: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

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

        # 1. Gather ONLY Positive samples in the batch for Softmax calculation
        # Softmax requires (User, PosItem, NegItems...) structure.
        # In general pointwise loader, we have mixed pos/neg samples.
        # Strategy: Use only (U, I) pairs where label=1 as "Anchors" for Softmax.
        
        mask_pos = (labels == 1)
        if not mask_pos.any():
            # Fallback if no positive samples in batch (rare but possible)
            return super().forward_pointwise(uids, iids, labels, pos_weight, sample_weight, test, lambda_reg)

        # Filter active positive pairs
        u_emb_pos = E_u[uids[mask_pos]]  # (N_pos, d)
        i_emb_pos = E_i[iids[mask_pos]]  # (N_pos, d)
        
        # 2. Similarity Matrix: (N_pos, N_pos)
        # Row i corresponds to user u_i.
        # Col j corresponds to item i_j.
        # Diagonal elements (i, i) are True Positives.
        # Off-diagonal elements (i, j) are In-batch Negatives (other items).
        
        sim_matrix = torch.matmul(u_emb_pos, i_emb_pos.T) / self.temperature
        
        # Label: diagonal indices [0, 1, 2, ..., N_pos-1]
        targets = torch.arange(u_emb_pos.size(0), device=u_emb_pos.device)
        
        loss_softmax = self.ce_loss(sim_matrix, targets)

        if sample_weight is not None:
             # Match weights to positive samples
             w_pos = sample_weight[mask_pos]
             loss_softmax = loss_softmax * w_pos

        loss_reg = (u_emb_pos.norm(2).pow(2) + i_emb_pos.norm(2).pow(2)).mean()
        loss = loss_softmax.mean() + lambda_reg * loss_reg

        # For metric logging, we still need logits for ALL samples
        u_emb_all = E_u[uids]
        i_emb_all = E_i[iids]
        logits = (u_emb_all * i_emb_all).sum(-1)

        return loss, logits


class CoPLGCFMargin(CoPLGCF):
    """
    CoPL-GCF adapted for Pairwise Margin Ranking Loss.
    
    Goal: Maximize separation between Positive and Negative scores.
    Loss = max(0, margin - (s_pos - s_neg))
    
    Features:
    - Does NOT force scores to specific targets (0/1).
    - Only cares about the RELATIVE order (gap) between positive and negative items.
    - Matches the intuition of "Separating Pos/Neg" without ground-truth value constraint.
    - Requires mixed Pos/Neg samples in each batch (handled by WeightedRandomSampler).
    """
    def __init__(self, *args, margin: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = float(margin)

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
        scores = (u_emb * i_emb).sum(-1)  # (B,)

        # 1. Identify Pos/Neg in batch
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_scores = scores[pos_mask]  # (N_pos,)
        neg_scores = scores[neg_mask]  # (N_neg,)
        
        # 2. Regularization (L2 norm)
        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        
        # 3. Pairwise Margin Loss
        # Broadcast to form all pairs: (N_pos, N_neg)
        # diff[i, j] = pos_scores[i] - neg_scores[j]
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        
        bal_weight = len(neg_scores) / (len(pos_scores) + 1e-6) 
        loss_margin = (-F.logsigmoid(diff) * bal_weight).mean()

        # (Experiment) Hinge Margin Loss
        # Loss = mean( max(0, margin - diff) )
        # loss_margin = F.relu(self.margin - diff).mean()

        # (Experiment) BPR Loss (LogSigmoid): Matches Eq(5) in paper
        # loss_margin = -F.logsigmoid(diff).mean()

        # (Experiment 1) Hard Negative Mining: Focus on the hardest negative (smallest diff) per positive
        # min_diff, _ = diff.min(dim=1)  # (N_pos,)
        # loss_margin = F.relu(self.margin - min_diff).mean()

        # (Experiment 4) Listwise Softmax (InfoNCE): Treat all negatives in batch as contrastive samples
        # tau = 0.1
        # # Concatenate [Pos, Neg1, Neg2, ...] for each positive
        # logits = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(0).repeat(len(pos_scores), 1)], dim=1) / tau
        # # Target is always index 0 (the positive sample)
        # list_labels = torch.zeros(len(pos_scores), dtype=torch.long, device=logits.device)
        # loss_margin = F.cross_entropy(logits, list_labels)
        
        loss = loss_margin + lambda_reg * loss_reg

        return loss, scores
