import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseModel import GeneralModel

class ANS(GeneralModel):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['embedding_size']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--embedding_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--gama', type=int, default=0.03,
							help='Used to adjust the impact of the contrastive loss and the disentanglement loss.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.embedding_size = args.embedding_size
		self.gama = args.gama
		self.num_neg = args.num_neg
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items

		self._define_params()
		self.apply(self.init_weights)


	def _define_params(self):
		# 定义嵌入层
		self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
		self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)

	def forward(self, feed_dict):
		user_ids = feed_dict['user_id']  # [batch_size]
		item_ids = feed_dict['item_id']  # [batch_size, -1]

		user_emb = self.user_embedding(user_ids)
		item_emb = self.item_embedding(item_ids)

		prediction = (user_emb[:, None, :] * item_emb).sum(dim=-1)  # [batch_size, -1]
		user_emb = user_emb.repeat(1,item_ids.shape[1]).view(item_ids.shape[0],item_ids.shape[1],-1)
		item_emb = item_emb
		out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1), 
			  		'u_v': user_emb, 
			  		'i_v': item_emb	}
		return out_dict

	def bpr_loss(self, pos_score, neg_score):
		return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))

	def contrastive_loss(self, pos_emb, neg_emb):
		pos_distance = torch.norm(pos_emb - neg_emb, p=2, dim=1)
		neg_distance = torch.clamp(1 - pos_distance, min=0)
		return torch.mean(neg_distance)

	def disentanglement_loss(self, user_emb, item_emb):
		user_norm = torch.norm(user_emb, p=2, dim=1)
		item_norm = torch.norm(item_emb, p=2, dim=1)
		return torch.mean(user_norm + item_norm)

	def loss(self, out_dict):
		prediction = out_dict['prediction']
		pos_score, neg_score = prediction[:, 0], prediction[:, 1:]
		pos_user_emb, pos_item_emb, neg_item_emb = out_dict['u_v'][:, 0], out_dict['i_v'][:, 0], out_dict['i_v'][:, 1:]

		bpr_loss = self.bpr_loss(pos_score, neg_score)
		contrastive_loss = self.contrastive_loss(pos_item_emb, neg_item_emb)
		dis_loss = self.disentanglement_loss(pos_user_emb, pos_item_emb)
		loss = bpr_loss + self.gama * (contrastive_loss + dis_loss)
		return loss
	
	def generate_augmented_negative_samples(self, user_id, neg_items, top_k=10):
		user_emb = self.user_embedding(user_id)
		item_embs = self.item_embedding(neg_items)  # 获取所有负样本的嵌入向量
		similarities = torch.cosine_similarity(user_emb, item_embs, dim=-1)

		# 选择相似度最高的k个负样本
		_, top_k_indices = torch.topk(similarities, top_k)
		augmented_neg_items = neg_items[top_k_indices.cpu().numpy()]

		return augmented_neg_items

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
