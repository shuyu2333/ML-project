# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from models.BaseModel import GeneralModel

class ANSLightGCN(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCN layers.')
        parser.add_argument('--gamma', type=float, default=0.03,
                            help='Weight for adversarial loss components.')
        parser.add_argument('--eps', type=float, default=0.1,
                            help='Weight for synthetic negative samples.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.gamma = args.gamma
        self.eps = args.eps
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        
        # 构建归一化邻接矩阵
        self.norm_adj = self._build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        
        # 定义模型参数
        self._define_params()
        self.apply(self.init_weights)

    @staticmethod
    def _build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1)) + 1e-10
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)

        return norm_adj_mat.tocsr()

    def _define_params(self):
        # 嵌入层
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)
        
        # 门控网络用于对抗训练
        self.user_gate = nn.Linear(self.emb_size, self.emb_size)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size)
        self.margin_model = nn.Linear(self.emb_size, 1)
        
        # 将scipy稀疏矩阵转换为PyTorch张量
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        # torch.sparse.FloatTensor is deprecated; use sparse_coo_tensor
        return torch.sparse_coo_tensor(i, v, coo.shape, dtype=torch.float32)

    def _aggregate_embeddings(self, users, items):
        """
        执行LightGCN的图卷积操作
        """
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # 堆叠并平均所有层的嵌入
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_num, :]
        item_all_embeddings = all_embeddings[self.user_num:, :]

        # ensure indices are integer long tensors on the same device
        if isinstance(users, np.ndarray):
            users_t = torch.from_numpy(users).long().to(user_all_embeddings.device)
        elif isinstance(users, torch.Tensor):
            users_t = users.long().to(user_all_embeddings.device)
        else:
            users_t = torch.tensor(users, dtype=torch.long, device=user_all_embeddings.device)

        # items may be 2D (batch_size, n_candidates)
        if isinstance(items, np.ndarray):
            items_t = torch.from_numpy(items).long().to(item_all_embeddings.device)
        elif isinstance(items, torch.Tensor):
            items_t = items.long().to(item_all_embeddings.device)
        else:
            items_t = torch.tensor(items, dtype=torch.long, device=item_all_embeddings.device)

        user_embeddings = user_all_embeddings[users_t, :]
        item_embeddings = item_all_embeddings[items_t, :]

        return user_embeddings, item_embeddings

    def forward(self, feed_dict):
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, -1]

        # 使用LightGCN聚合嵌入
        user_emb, item_emb = self._aggregate_embeddings(user_ids, item_ids)

        # 计算预测得分
        prediction = (user_emb[:, None, :] * item_emb).sum(dim=-1)  # [batch_size, -1]
        
        # 重塑输出
        user_emb_expanded = user_emb.repeat(1, item_ids.shape[1]).view(item_ids.shape[0], item_ids.shape[1], -1)
        item_emb_reshaped = item_emb
        
        out_dict = {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'u_v': user_emb_expanded,
            'i_v': item_emb_reshaped
        }
        return out_dict

    def bpr_loss(self, pos_score, neg_score):
        """标准BPR损失"""
        return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))

    def adversarial_negative_sampling_loss(self, u_emb, pos_emb, neg_emb):
        """
        对抗负采样损失，包括hard/easy样本分离和正交性约束
        """
        batch_size = u_emb.shape[0]
        
        # 计算hard负样本门控
        gate_neg_hard = torch.sigmoid(
            self.item_gate(neg_emb) * self.user_gate(u_emb).unsqueeze(1)
        )
        
        # 分离hard和easy负样本
        n_hard = neg_emb * gate_neg_hard
        n_easy = neg_emb - n_hard
        p_hard = pos_emb.unsqueeze(1) * gate_neg_hard
        p_easy = pos_emb.unsqueeze(1) - p_hard
        
        # 计算距离损失
        distance = torch.mean(torch.pairwise_distance(n_hard, p_hard, p=2).squeeze(dim=1))
        
        # 正交性约束
        temp = torch.norm(torch.mul(p_easy, n_easy), dim=-1)
        orth = torch.mean(torch.sum(temp, axis=-1))
        
        # 边际模型
        margin = torch.sigmoid(1 / self.margin_model(n_hard * p_hard))

        # 生成合成负样本
        random_noise = torch.rand(n_easy.shape).to(self.device)
        magnitude = torch.nn.functional.normalize(random_noise, p=2, dim=-1) * margin * 0.1
        direction = torch.sign(p_easy - n_easy)
        noise = torch.mul(direction, magnitude)
        n_easy_syth = noise + n_easy
        n_synthetic = n_hard + n_easy_syth
        
        # 计算不同类型负样本的得分
        hard_scores = torch.sum(torch.mul(u_emb.unsqueeze(dim=1), n_hard), axis=-1)
        easy_scores = torch.sum(torch.mul(u_emb.unsqueeze(dim=1), n_easy), axis=-1)
        synth_scores = torch.sum(torch.mul(u_emb.unsqueeze(dim=1), n_synthetic), axis=-1)
        norm_scores = torch.sum(torch.mul(u_emb.unsqueeze(dim=1), neg_emb), axis=-1)
        
        # SNS损失
        sns_loss = torch.mean(torch.log(1 + torch.exp(easy_scores - hard_scores).sum(dim=1)))
        
        # 距离损失
        dis_loss = distance + orth
        
        # 选择最优负样本
        scores = (u_emb.unsqueeze(dim=1) * n_synthetic).sum(dim=-1)
        scores_false = synth_scores - norm_scores
        indices = torch.max(scores + self.eps * scores_false, dim=1)[1].detach()
        
        # 重新排列嵌入以获取选定的负样本
        # select chosen negative embedding for each batch element
        # n_synthetic: [batch_size, num_neg, emb_dim]
        idx = torch.arange(batch_size, device=n_synthetic.device)
        indices = indices.long().to(n_synthetic.device)
        selected_neg_emb = n_synthetic[idx, indices, :]
        
        return sns_loss, dis_loss, selected_neg_emb

    def loss(self, out_dict):
        prediction = out_dict['prediction']
        pos_score, neg_score = prediction[:, 0], prediction[:, 1:]
        pos_user_emb, pos_item_emb = out_dict['u_v'][:, 0], out_dict['i_v'][:, 0]
        neg_item_emb = out_dict['i_v'][:, 1:]
        
        # 标准BPR损失
        bpr_loss = self.bpr_loss(pos_score, neg_score)
        
        # 对抗负采样损失
        sns_loss, dis_loss, selected_neg_emb = self.adversarial_negative_sampling_loss(
            pos_user_emb, pos_item_emb, neg_item_emb
        )
        
        # 组合损失
        total_loss = bpr_loss + self.gamma * (sns_loss + dis_loss)
        
        return total_loss

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)

        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict
