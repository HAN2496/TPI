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
                 dropout=0.1, l=3, item_item_adj_norm=None, item_item_weight=1.0,
                 loss_type="bce", loss_kwargs=None, item_feat_init=None):
        super().__init__()
        self.n_u, self.n_i, self.d = n_u, n_i, d
        self.l = l
        self.dropout = float(dropout)
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs

        self.pos_adj_norm = pos_adj_norm.coalesce().float()
        self.neg_adj_norm = neg_adj_norm.coalesce().float()
        self.item_item_adj_norm = None if item_item_adj_norm is None else item_item_adj_norm.coalesce().float()
        self.item_item_weight = float(item_item_weight)

        self.act = nn.LeakyReLU(0.1)

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))

        # ====== Random init (original) ======
        # self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # ====== VAE latent feature init ======
        self.E_i_0 = nn.Parameter(item_feat_init.clone().float())

        # ====== Original CoPL weights ======
        # self.W_u_pos  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # defined but unused in forward (original bug)
        # self.W_i_pos  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # used for hadamard of BOTH user & item pos
        # self.W_u_neg  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # defined but unused in forward (original bug)
        # self.W_i_neg  = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # used for hadamard of BOTH user & item neg
        # self.W_u_self = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # used for BOTH user & item self (original bug)
        # self.W_i_self = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])  # defined but unused in forward (original bug)

        # ====== Our corrected weights (commented out) ======
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

        self.W_i_cross_pos_user = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])
        self.W_i_cross_neg_user = nn.ModuleList([nn.Linear(d, d) for _ in range(l)])

        if loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss(margin=self.loss_kwargs["margin"], reduction="none")
        elif loss_type == "softmax":
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')

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

            # ====== Original CoPL aggregation ======
            # W_i_pos/W_i_neg used for both user & item hadamard (shared weights)
            # W_u_self used for both user & item self-connection (original bug)
            # m_u = (self.W_u_self[layer](E_u_prev) + 
            #        + self.W_i_pos[layer](Z_u_pos * E_u_prev) + Z_u_pos
            #        + self.W_i_neg[layer](Z_u_neg * E_u_prev) + Z_u_neg)

            # m_i = (Z_i_pos + self.W_i_pos[layer](Z_i_pos * E_i_prev)
            #        + Z_i_neg + self.W_i_neg[layer](Z_i_neg * E_i_prev)
            #        + self.W_i_self[layer](E_i_prev))  # original bug: W_u_self for items

            # ====== Our corrected aggregation (commented out) ======
            m_u = (self.W_u_self[layer](E_u_prev)
                   + self.W_u_pos_1[layer](Z_u_pos)
                   + self.W_u_pos_2[layer](Z_u_pos * E_u_prev)
                   + self.W_u_neg_3[layer](Z_u_neg)
                   + self.W_u_neg_4[layer](Z_u_neg * E_u_prev))
            m_i = self.W_i_self[layer](E_i_prev)
            # m_i = (self.W_i_self[layer](E_i_prev)
            #        + self.W_i_pos_1[layer](Z_i_pos)
            #        + self.W_i_pos_2[layer](Z_i_pos * E_i_prev)
            #        + self.W_i_neg_3[layer](Z_i_neg)
            #        + self.W_i_neg_4[layer](Z_i_neg * E_i_prev))

            if self.item_item_adj_norm is not None:
                Aii = sparse_dropout(self.item_item_adj_norm, self.dropout, training=not test)
                Z_i_ii_item = torch.spmm(Aii, E_i_prev)
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii_item) + self.W_i_ii_2[layer](Z_i_ii_item * E_i_prev))

                # 2. 유사한 궤적을 평가한 '타 유저'의 임베딩 흡수 (상상하셨던 핵심!)
                Z_i_ii_user = torch.spmm(Aii, Z_i_pos) 
                Z_i_ii_user_neg = torch.spmm(Aii, Z_i_neg) 

                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii_item) + 
                    self.W_i_ii_2[layer](Z_i_ii_item * E_i_prev) +
                    self.W_i_cross_pos_user[layer](Z_i_ii_user) +
                    self.W_i_cross_neg_user[layer](Z_i_ii_user_neg)
                )

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        if self.loss_type == "cosine":        
            self.E_u = F.normalize(E_u_prev, dim=-1)
            self.E_i = F.normalize(E_i_prev, dim=-1)
        else:
            self.E_i = E_i_prev
            self.E_u = E_u_prev
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
        scores = (u_emb * i_emb).sum(-1)
        loss_reg = (u_emb.pow(2).sum(dim=-1).mean() + i_emb.pow(2).sum(dim=-1).mean())

        if self.loss_type == "bce":
            loss = self.weighted_bce_with_logits(scores, labels, pos_weight, sample_weight)
            return loss + lambda_reg * loss_reg, scores

        elif self.loss_type == "bce_diversity":
            loss = self.weighted_bce_with_logits(scores, labels, pos_weight, sample_weight)
            U = self.E_u
            sim = (U @ U.T) / (U.norm(dim=1, keepdim=True) * U.norm(dim=1, keepdim=True).T + 1e-8)
            user_div_loss = sim.triu(diagonal=1).mean() 
            return loss + lambda_reg * loss_reg + self.loss_kwargs["lambda_div"] * user_div_loss, scores

        elif self.loss_type == "bpr":
            log_prob_pos = F.logsigmoid(scores)
            log_prob_neg = F.logsigmoid(-scores)
            if pos_weight is not None:
                loss = -(labels * log_prob_pos * pos_weight + (1 - labels) * log_prob_neg)
            else:
                loss = -(labels * log_prob_pos + (1 - labels) * log_prob_neg)
            if sample_weight is not None:
                loss = loss * sample_weight
            return loss.mean() + lambda_reg * loss_reg, scores

        elif self.loss_type == "cosine":
            target = labels.float().clone()
            target[target == 0] = -1
            loss = self.loss_fn(u_emb, i_emb, target)
            if sample_weight is not None:
                loss = loss * sample_weight
            return loss.mean(), scores

        elif self.loss_type == "softmax":
            mask_pos = (labels == 1)
            if not mask_pos.any():
                loss = self.weighted_bce_with_logits(scores, labels, pos_weight, sample_weight)
                return loss + lambda_reg * loss_reg, scores
            u_pos = u_emb[mask_pos]
            i_pos = i_emb[mask_pos]
            sim_matrix = torch.matmul(u_pos, i_pos.T) / self.loss_kwargs["temperature"]
            targets = torch.arange(u_pos.size(0), device=u_pos.device)
            loss_ce = self.ce_loss(sim_matrix, targets)
            if sample_weight is not None:
                loss_ce = loss_ce * sample_weight[mask_pos]
            loss_reg = (u_pos.pow(2).sum(dim=-1).mean() + i_pos.pow(2).sum(dim=-1).mean())
            return loss_ce.mean() + lambda_reg * loss_reg, scores

        elif self.loss_type == "margin":
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            bal_weight = len(neg_scores) / (len(pos_scores) + 1e-6)
            loss = (-F.logsigmoid(diff) * bal_weight).mean()
            return loss + lambda_reg * loss_reg, scores

        raise ValueError(f"Unknown loss_type: {self.loss_type}")
