import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(mat: torch.Tensor, p: float, training: bool):
    if (not training) or p <= 0.0:
        return mat
    mat = mat.coalesce()
    idx = mat.indices()
    val = F.dropout(mat.values(), p=p, training=True)
    return torch.sparse_coo_tensor(idx, val, mat.size(), device=mat.device).coalesce()


class CoPLGCF(nn.Module):
    def __init__(self, n_u, n_i, d, pos_adj_norm, neg_adj_norm,
                 dropout=0.1, l=3, item_item_adj_norm=None, item_item_weight=1.0):
        super().__init__()
        self.n_u, self.n_i, self.d = n_u, n_i, d
        self.l = l
        self.dropout = float(dropout)

        self.pos_adj_norm = pos_adj_norm.coalesce().float()
        self.neg_adj_norm = neg_adj_norm.coalesce().float()
        self.item_item_adj_norm = None if item_item_adj_norm is None else item_item_adj_norm.coalesce().float()
        self.item_item_weight = float(item_item_weight)

        self.act = nn.LeakyReLU(0.2)

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        self.W_u_self  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_u_pos_1 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_u_pos_2 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_u_neg_3 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_u_neg_4 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])

        self.W_i_self  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_pos_1 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_pos_2 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_neg_3 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_neg_4 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])

        self.W_i_ii_1 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_ii_2 = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])

        self.E_u = None
        self.E_i = None

    def encode_graph(self, test=False):
        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)

            Z_u_pos = torch.spmm(Apos, E_i_prev)
            Z_u_neg = torch.spmm(Aneg, E_i_prev)
            Z_i_pos = torch.spmm(Apos.transpose(0, 1), E_u_prev)
            Z_i_neg = torch.spmm(Aneg.transpose(0, 1), E_u_prev)

            m_u = (self.W_u_self[layer](E_u_prev)
                   + self.W_u_pos_1[layer](Z_u_pos)
                   + self.W_u_pos_2[layer](Z_u_pos * E_u_prev)
                   + self.W_u_neg_3[layer](Z_u_neg)
                   + self.W_u_neg_4[layer](Z_u_neg * E_u_prev))

            m_i = (self.W_i_self[layer](E_i_prev)
                   + self.W_i_pos_1[layer](Z_i_pos)
                   + self.W_i_pos_2[layer](Z_i_pos * E_i_prev)
                   + self.W_i_neg_3[layer](Z_i_neg)
                   + self.W_i_neg_4[layer](Z_i_neg * E_i_prev))

            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                Z_i_ii = torch.spmm(Aii, E_i_prev)
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii) + self.W_i_ii_2[layer](Z_i_ii * E_i_prev))

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        self.E_u = F.normalize(E_u_prev, dim=-1)
        self.E_i = E_i_prev
        return self.E_u, self.E_i

    @staticmethod
    def weighted_bce_with_logits(logits, labels, pos_weight=None, sample_weight=None):
        labels = labels.float()
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def forward_pointwise(self, uids, iids, labels, pos_weight=None, sample_weight=None,
                          test=False, lambda_reg=1e-6):
        E_u, E_i = self.encode_graph(test=test)
        u_emb = E_u[uids]
        i_emb = E_i[iids]
        logits = (u_emb * i_emb).sum(-1)
        loss_bce = self.weighted_bce_with_logits(logits, labels, pos_weight, sample_weight)
        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        return loss_bce + lambda_reg * loss_reg, logits


class CoPLGCFCosine(CoPLGCF):
    def __init__(self, *args, margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="none")

    def encode_graph(self, test=False):
        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)

            Z_u_pos = torch.spmm(Apos, E_i_prev)
            Z_u_neg = torch.spmm(Aneg, E_i_prev)
            Z_i_pos = torch.spmm(Apos.transpose(0, 1), E_u_prev)
            Z_i_neg = torch.spmm(Aneg.transpose(0, 1), E_u_prev)

            m_u = (self.W_u_self[layer](E_u_prev)
                   + self.W_u_pos_1[layer](Z_u_pos)
                   + self.W_u_pos_2[layer](Z_u_pos * E_u_prev)
                   + self.W_u_neg_3[layer](Z_u_neg)
                   + self.W_u_neg_4[layer](Z_u_neg * E_u_prev))

            m_i = (self.W_i_self[layer](E_i_prev)
                   + self.W_i_pos_1[layer](Z_i_pos)
                   + self.W_i_pos_2[layer](Z_i_pos * E_i_prev)
                   + self.W_i_neg_3[layer](Z_i_neg)
                   + self.W_i_neg_4[layer](Z_i_neg * E_i_prev))

            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                Z_i_ii = torch.spmm(Aii, E_i_prev)
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii) + self.W_i_ii_2[layer](Z_i_ii * E_i_prev))

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        self.E_u = F.normalize(E_u_prev, dim=-1)
        self.E_i = F.normalize(E_i_prev, dim=-1)
        return self.E_u, self.E_i

    def forward_pointwise(self, uids, iids, labels, pos_weight=None, sample_weight=None,
                          test=False, lambda_reg=1e-6):
        E_u, E_i = self.encode_graph(test=test)
        u_emb = E_u[uids]
        i_emb = E_i[iids]
        scores = (u_emb * i_emb).sum(-1)
        target = labels.float().clone()
        target[target == 0] = -1
        loss = self.loss_fn(u_emb, i_emb, target)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean(), scores


class CoPLGCFPointwiseBPR(CoPLGCF):
    def forward_pointwise(self, uids, iids, labels, pos_weight=None, sample_weight=None,
                          test=False, lambda_reg=1e-6):
        E_u, E_i = self.encode_graph(test=test)
        u_emb = E_u[uids]
        i_emb = E_i[iids]
        logits = (u_emb * i_emb).sum(-1)

        log_prob_pos = F.logsigmoid(logits)
        log_prob_neg = F.logsigmoid(-logits)

        if pos_weight is not None:
            loss = -(labels * log_prob_pos * pos_weight + (1 - labels) * log_prob_neg)
        else:
            loss = -(labels * log_prob_pos + (1 - labels) * log_prob_neg)

        if sample_weight is not None:
            loss = loss * sample_weight

        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        return loss.mean() + lambda_reg * loss_reg, logits


class CoPLGCFSoftmax(CoPLGCF):
    def __init__(self, *args, temperature=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward_pointwise(self, uids, iids, labels, pos_weight=None, sample_weight=None,
                          test=False, lambda_reg=1e-6):
        E_u, E_i = self.encode_graph(test=test)
        mask_pos = (labels == 1)
        if not mask_pos.any():
            return super().forward_pointwise(uids, iids, labels, pos_weight, sample_weight, test, lambda_reg)

        u_emb_pos = E_u[uids[mask_pos]]
        i_emb_pos = E_i[iids[mask_pos]]
        sim_matrix = torch.matmul(u_emb_pos, i_emb_pos.T) / self.temperature
        targets = torch.arange(u_emb_pos.size(0), device=u_emb_pos.device)
        loss_softmax = self.ce_loss(sim_matrix, targets)

        if sample_weight is not None:
            loss_softmax = loss_softmax * sample_weight[mask_pos]

        loss_reg = (u_emb_pos.norm(2).pow(2) + i_emb_pos.norm(2).pow(2)).mean()
        u_emb_all = E_u[uids]
        i_emb_all = E_i[iids]
        logits = (u_emb_all * i_emb_all).sum(-1)
        return loss_softmax.mean() + lambda_reg * loss_reg, logits


class CoPLGCFMargin(CoPLGCF):
    def __init__(self, *args, margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = float(margin)

    def forward_pointwise(self, uids, iids, labels, pos_weight=None, sample_weight=None,
                          test=False, lambda_reg=1e-6):
        E_u, E_i = self.encode_graph(test=test)
        u_emb = E_u[uids]
        i_emb = E_i[iids]
        scores = (u_emb * i_emb).sum(-1)

        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        loss_reg = (u_emb.norm(2).pow(2) + i_emb.norm(2).pow(2)).mean()
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        bal_weight = len(neg_scores) / (len(pos_scores) + 1e-6)
        loss_margin = (-F.logsigmoid(diff) * bal_weight).mean()

        return loss_margin + lambda_reg * loss_reg, scores
