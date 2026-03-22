import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcf import sparse_dropout


class CoPLGCF_GCN(nn.Module):
    def __init__(self, n_u, n_i, d, pos_adj_norm, neg_adj_norm,
                 dropout=0.1, l=3, item_item_adj_norm=None, item_item_weight=1.0):
        super().__init__()
        self.n_u, self.n_i, self.d = n_u, n_i, d
        self.l = l
        self.dropout = float(dropout)
        self.item_item_weight = float(item_item_weight)

        self.register_buffer("pos_adj_norm", pos_adj_norm.coalesce().float())
        self.register_buffer("neg_adj_norm", neg_adj_norm.coalesce().float())
        self.register_buffer("item_item_adj_norm", item_item_adj_norm.coalesce().float())

        self.act = nn.LeakyReLU(0.2)
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(l)])

    def encode_graph(self, test=False):
        E_u = self.E_u_0
        E_i = self.E_i_0

        for linear in self.layers:
            Apos = sparse_dropout(self.pos_adj_norm, self.dropout, training=not test)
            Aneg = sparse_dropout(self.neg_adj_norm, self.dropout, training=not test)

            msg_u = torch.spmm(Apos, E_i) - torch.spmm(Aneg, E_i)
            msg_i = torch.spmm(Apos.transpose(0, 1), E_u) - torch.spmm(Aneg.transpose(0, 1), E_u)

            Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
            msg_i = msg_i + self.item_item_weight * torch.spmm(Aii, E_i)

            E_u = self.act(linear(msg_u + E_u))
            E_i = self.act(linear(msg_i + E_i))

        return F.normalize(E_u, dim=-1), E_i

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
