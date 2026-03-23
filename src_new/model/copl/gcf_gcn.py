import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class SparseAggregator(MessagePassing):
    """단순 덧셈 집계(Aggregation)를 수행하는 커스텀 PyG 레이어"""
    def __init__(self, dropout=0.0):
        super().__init__(aggr='add')
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # 학습 시 엣지 드롭아웃 적용 (Graph 엣지에 노이즈 주입)
        if self.training and self.dropout > 0.0 and edge_weight is not None:
            edge_weight = F.dropout(edge_weight, p=self.dropout, training=True)
        # x는 bipartite 그래프를 위해 (x_source, x_target) 튜플 형태로 입력받음
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class CoPLGCF_PyG(nn.Module):
    def __init__(self, n_u, n_i, d, l=3, dropout=0.1, item_item_weight=1.0):
        super().__init__()
        self.n_u, self.n_i, self.d = n_u, n_i, d
        self.l = l
        self.item_item_weight = float(item_item_weight)

        self.act = nn.LeakyReLU(0.2)
        
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        self.aggregator = SparseAggregator(dropout=float(dropout))

        # 독립적인 가중치 파라미터들
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

    def encode_graph(self, edge_index_dict, edge_weight_dict=None):
        if edge_weight_dict is None:
            edge_weight_dict = {}

        E_u_prev = self.E_u_0
        E_i_prev = self.E_i_0

        for layer in range(self.l):
            # 1. Bipartite Graph 집계 (Item -> User 방향)
            Z_u_pos = self.aggregator(
                (E_i_prev, E_u_prev), edge_index_dict['i2u_pos'], edge_weight_dict.get('i2u_pos'))
            Z_u_neg = self.aggregator(
                (E_i_prev, E_u_prev), edge_index_dict['i2u_neg'], edge_weight_dict.get('i2u_neg'))

            # 2. Bipartite Graph 집계 (User -> Item 방향)
            Z_i_pos = self.aggregator(
                (E_u_prev, E_i_prev), edge_index_dict['u2i_pos'], edge_weight_dict.get('u2i_pos'))
            Z_i_neg = self.aggregator(
                (E_u_prev, E_i_prev), edge_index_dict['u2i_neg'], edge_weight_dict.get('u2i_neg'))

            # 3. User & Item 임베딩 업데이트
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

            # 4. Item-Item Graph 집계 (Item -> Item 방향)
            if 'i2i' in edge_index_dict:
                Z_i_ii = self.aggregator(
                    (E_i_prev, E_i_prev), edge_index_dict['i2i'], edge_weight_dict.get('i2i'))
                m_i = m_i + self.item_item_weight * (
                    self.W_i_ii_1[layer](Z_i_ii) + self.W_i_ii_2[layer](Z_i_ii * E_i_prev))

            E_u_prev = self.act(m_u)
            E_i_prev = self.act(m_i)

        return F.normalize(E_u_prev, dim=-1), E_i_prev

    def forward_pointwise(self, uids, iids, labels, edge_index_dict, edge_weight_dict=None,
                          pos_weight=None, sample_weight=None, lambda_reg=1e-6):
        # 정방향 전파로 전체 노드 임베딩 추출
        E_u, E_i = self.encode_graph(edge_index_dict, edge_weight_dict)
        
        u_emb = E_u[uids]
        i_emb = E_i[iids]
        
        logits = (u_emb * i_emb).sum(-1)
        
        # Loss 계산
        loss_bce = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight, reduction="none"
        )
        if sample_weight is not None:
            loss_bce = loss_bce * sample_weight
        loss_bce = loss_bce.mean()
        
        # 앞서 최적화한 L2 정규화 방식 적용
        loss_reg = (u_emb.pow(2).sum(dim=-1).mean() + i_emb.pow(2).sum(dim=-1).mean())
        
        return loss_bce + lambda_reg * loss_reg, logits