import sys
sys.path.insert(0, 'src')
import argparse, torch
from helpers.BaseReader import BaseReader
from models.ANS import ANS

args = argparse.Namespace(path='../data/', dataset='Grocery_and_Gourmet_Food', sep='\t', device=torch.device('cpu'), model_path='', buffer=0, num_neg=1, embedding_size=64, gama=0.03, num_workers=0, eval_batch_size=8, train=0, test_all=0)
corpus = BaseReader(args)
model = ANS(args, corpus)
dataset = model.Dataset(model, corpus, 'test')
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_batch, num_workers=0)
batch = next(iter(loader))
print('Loaded batch keys:', list(batch.keys()))
print('item_id dtype:', batch['item_id'].dtype, 'shape:', batch['item_id'].shape)