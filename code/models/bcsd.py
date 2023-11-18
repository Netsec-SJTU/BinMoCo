import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import lightning as L
import torch
from constants import max_block_token_len
from models.gnns import GatedGCN
from models.knn import FaissKNN
from models.metrics import RankingMetrics
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.loss_and_miner_utils import (
    get_all_triplets_indices, get_random_triplet_indices)
from torch import nn
from torch_geometric.nn import GIN, AttentionalAggregation
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup


class DmlBcsdBaseModule(L.LightningModule):
    def __init__(self, lr, milestones):
        super(DmlBcsdBaseModule, self).__init__()
        self.lr = lr
        self.milestones = milestones
        
    def forward(self, data):
        raise NotImplementedError

    def cal_metrics(self, x, gid):
        _, knn_indexes = FaissKNN(use_gpu=False)(query=x, reference=x, k=x.shape[0])
        metrics = RankingMetrics()(knn_indexes, gid, gid, same_source=True)
        return metrics

    def validation_step(self, data, *args):
        x = self(data)
        val_metrics = self.cal_metrics(x, data.gid)
        val_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def test_step(self, data, *args):
        x = self(data)
        test_metrics = self.cal_metrics(x, data.gid)
        test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
        self.log_dict(test_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        # ! for BAR2019 model, remove embedding layer from optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        if self.milestones is None:
            stepping_batches = self.trainer.estimated_stepping_batches
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.15*stepping_batches),
                num_training_steps=stepping_batches
            )
            print(f'using linear-warmup scheduler. total steps: {stepping_batches}.')
            return [optimizer], [{'scheduler': lr_scheduler,
                                  'interval': 'step',
                                  'frequency': 1,
                                  'name': 'linear_warmup'}]
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones)
            return [optimizer], [scheduler]


class Transformer_GNN_CG_GNN(DmlBcsdBaseModule):
    def __init__(
        self,
        num_edge_type=3,
        embedding_dims=128,
        trans_hidden_dims=256,
        trans_layers=6,
        gnn_name='gatedgcn',
        gnn_hidden_dims=128,
        gnn_out_dims=128,
        gnn_layers=3,
        dropout=0.5,
        vocab_size=5864,
        type_vocab_size=61,
        imp_vocab_size=1000,
        lr=1e-3,
        milestones=[30, 60],
        miner_type='ms',
        loss_type='ms'
    ):
        super(Transformer_GNN_CG_GNN, self).__init__(lr, milestones)
        
        assert miner_type in ('ms', 'random', 'all')
        assert loss_type in ('ms', 'triplet', 'circle', 'contrastive')
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dims, padding_idx=0)
        self.token_type_embedding = nn.Embedding(type_vocab_size, embedding_dims, padding_idx=0)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=trans_hidden_dims,
            num_hidden_layers=trans_layers,
            num_attention_heads=4,
            intermediate_size=trans_hidden_dims*2,
            max_position_embeddings=max_block_token_len+1 # +1 for [CLS]
        )
        
        self.trans_encoder = BertModel(bert_config)
        
        self.edge_embedding = nn.Embedding(num_edge_type, embedding_dims)
        gnn_in_dims = trans_hidden_dims
        if gnn_name == 'gatedgcn':
            self.gnn = GatedGCN(
                in_channels=gnn_in_dims, 
                hidden_channels=gnn_hidden_dims, 
                num_layers=gnn_layers,
                out_channels=gnn_out_dims, 
                dropout=dropout, 
                jk='cat', 
                edge_dims=embedding_dims
            )
        self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))

        self.in_degree_embedding = nn.Embedding(21, gnn_hidden_dims)
        self.out_degree_embedding = nn.Embedding(21, gnn_hidden_dims)
        self.imp_embedding = nn.Embedding(imp_vocab_size, gnn_hidden_dims)
        self.layer_norm = nn.LayerNorm(gnn_hidden_dims)
        
        self.gnn_cg = GIN(
            in_channels=gnn_hidden_dims, 
            hidden_channels=gnn_hidden_dims,
            num_layers=1,
            out_channels=gnn_out_dims, 
            dropout=dropout, jk='cat'
        )
        self.pooling_cg = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))
        
        self.fc = nn.Linear(gnn_out_dims*2, gnn_out_dims)
        
        self.miner_type = miner_type
        self.loss_type = loss_type
        print(f'Miner: {self.miner_type}. Loss: {self.loss_type}.')

        # miner        
        if miner_type == 'ms':
            self.miner = miners.MultiSimilarityMiner()
        elif miner_type == 'random':
            self.miner = get_random_triplet_indices
        elif miner_type == 'all':
            self.miner = get_all_triplets_indices
        else:
            raise Exception(f'invalid loss type {loss_type}.')
        
        # loss
        if loss_type == 'ms':
            self.criterion = losses.MultiSimilarityLoss()
        elif loss_type == 'triplet':
            self.criterion = losses.TripletMarginLoss(margin=0.2)
        elif loss_type == 'circle':
            self.criterion = losses.CircleLoss()
        elif loss_type == 'contrastive':
            self.criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        else:
            raise Exception(f'invalid loss type {loss_type}.')

    def forward(self, data):
        x, x_type, padding_mask = data.x_cfg.to(torch.long), data.x_type.to(torch.long), data.padding_masks.to(torch.bool)
        edge_index, edge_type = data.edge_index_cfg.to(torch.long), data.edge_type.to(torch.long)
        batch = data.x_cfg_batch
        
        # * CFG
        # transformer input
        x = self.token_embedding(x) + self.token_type_embedding(x_type)
        
        # transformer
        x = self.trans_encoder(inputs_embeds=x, attention_mask=~padding_mask).pooler_output

        edge_type = self.edge_embedding(edge_type)
        x = self.gnn(x, edge_index, edge_attr=edge_type)
        x = self.pooling(x, batch)
        
        # * CG
        x_cg = data.x_cg.to(torch.long) # [node_num, 2]
        imp_embedding = self.imp_embedding(x_cg[:, 0])
        in_degree_embedding = self.in_degree_embedding(x_cg[:, 1])
        out_degree_embedding = self.out_degree_embedding(x_cg[:, 2])
        x_cg = imp_embedding + in_degree_embedding + out_degree_embedding
        x_cg = self.layer_norm(x_cg)
        
        edge_index_cg = data.edge_index_cg.to(torch.long)
        batch_cg = data.x_cg_batch
        x_cg = self.gnn_cg(x_cg, edge_index_cg)
        x_cg = self.pooling_cg(x_cg, batch_cg)
        
        # * concat cfg feat and cg feat
        x = torch.cat([x, x_cg], dim=1)
        
        # * important: Linear Norm head
        # * brings huge accuracy improvement
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
        
    def training_step(self, data, *args):
        x = self(data)
        # multi-similarity miner
        if self.miner_type == 'ms':
            apn_indices_tuple = self.miner(x, data.gid)
        # random miner
        elif self.miner_type == 'random':
            apn_indices_tuple = self.miner(data.gid, t_per_anchor=1)
        else: # all miner
            apn_indices_tuple = self.miner(data.gid)
        
        if apn_indices_tuple[0].shape[0] == 0 and apn_indices_tuple[2].shape[0] == 0:
            loss = torch.tensor(0.0, requires_grad=True,device=self.device)
        else:
            loss = self.criterion(embeddings=x, indices_tuple=apn_indices_tuple)
        self.log_dict({'train/loss': loss}, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return {'loss': loss}


class MomentumEncoderWrapper(DmlBcsdBaseModule):
    def __init__(
        self, 
        encQ,
        encK,
        out_dims,
        memory_size=4096,
        momentum=0.99,
        miner_type='ms',
        loss_type='ms',
        lr=1e-3,
        milestones=[30, 60]
    ):
        print('Training with Moco (Momentem Encoder).')
        super(MomentumEncoderWrapper, self).__init__(lr, milestones)
        
        assert miner_type in ('ms', 'random', 'all')
        assert loss_type in ('ms', 'triplet', 'circle', 'contrastive')
        
        self.encQ = encQ
        self.encK = encK
        self.memory_size = memory_size
        self.momentum = momentum
        
        # miner        
        if miner_type == 'ms':
            self.miner = miners.MultiSimilarityMiner()
        elif miner_type == 'random':
            self.miner = get_random_triplet_indices
        elif miner_type == 'all':
            self.miner = get_all_triplets_indices
        else:
            raise Exception(f'invalid loss type {loss_type}.')
        
        # loss
        if loss_type == 'ms':
            self.criterion = losses.MultiSimilarityLoss()
        elif loss_type == 'triplet':
            self.criterion = losses.TripletMarginLoss()
        elif loss_type == 'circle':
            self.criterion = losses.CircleLoss()
        elif loss_type == 'contrastive':
            self.criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        else:
            raise Exception(f'invalid loss type {loss_type}.')
        
        # cross batch memory
        if miner_type == 'all':
            self.xbm_criterion = losses.CrossBatchMemory(
                loss=self.criterion,
                embedding_size=out_dims,
                memory_size=self.memory_size,
                miner=None
            )
        else:
            self.xbm_criterion = losses.CrossBatchMemory(
                loss=self.criterion,
                embedding_size=out_dims,
                memory_size=self.memory_size,
                miner=self.miner
            )
       
        # init k with q
        self._momentum_update_key_encoder(momentum=None)

    def __forward_q(self, data):
        return self.encQ(data)
    
    def __forward_k(self, data):
        return self.encK(data)
    
    def forward(self, data):
        return self.__forward_q(data)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self, momentum=None):
        """from MoCo repo: momentum update of the key encoder"""
        # initialize before training
        if momentum is None:
            for param_q, param_k in zip(self.encQ.parameters(), self.encK.parameters()):
                param_k.data.copy_(param_q.data) 
                param_k.requires_grad = False # not update by gradient
        # momentum update
        else:
            for param_q, param_k in zip(self.encQ.parameters(), self.encK.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    # def on_train_epoch_start(self):
    #     """called on every train epoch start"""
    #     self.xbm_criterion.reset_queue()
    #     with torch.no_grad():
    #         self._momentum_update_key_encoder(momentum=None)
    
    def training_step(self, data, *args):
        self.encQ.train()
        self.encK.eval()
        x_q = self(data)

        with torch.no_grad():
            self._momentum_update_key_encoder(momentum=self.momentum)
            x_k = self.__forward_k(data)
        
        # concat two embeddings and construct enqueue_mask
        b, _ = x_q.shape
        x_all = torch.cat([x_q, x_k], dim=0)
        labels_all = torch.cat([data.gid, data.gid], dim=0).long()
        enqueue_mask = torch.zeros(len(labels_all)).bool()
        enqueue_mask[b:] = True
        
        loss = self.xbm_criterion(
            embeddings=x_all, labels=labels_all, enqueue_mask=enqueue_mask)
        self.log_dict({'train/loss': loss}, on_step=True, on_epoch=True, prog_bar=True, batch_size=x_q.size(0))
        return {'loss': loss}

    def configure_optimizers(self):
        # only optimize encQ
        optimizer = torch.optim.AdamW(self.encQ.parameters(), lr=self.lr)
        if self.milestones is None:
            stepping_batches = self.trainer.estimated_stepping_batches
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.15*stepping_batches),
                num_training_steps=stepping_batches
            )
            print(f'using linear-warmup scheduler. total steps: {stepping_batches}.')
            return [optimizer], [{'scheduler': lr_scheduler,
                                  'interval': 'step',
                                  'frequency': 1,
                                  'name': 'linear_warmup'}]
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones)
            return [optimizer], [scheduler]

