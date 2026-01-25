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
