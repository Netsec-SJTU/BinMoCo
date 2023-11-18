import argparse
import copy
import json
import os
import time

import lightning as L
from constants import seed
from dataset.pyg_data import CFG_CG_Data, CFGCGSimDataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger
from models.bcsd import MomentumEncoderWrapper, Transformer_GNN_CG_GNN
from pytorch_metric_learning import samplers
from torch_geometric.loader import DataLoader as pygDataLoader


def main():

    # seed
    L.seed_everything(seed)
    
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='database/binkit_2.0',
                        help='path of data directory')
    parser.add_argument('--data_repr', type=str, choices=('cfg_cg'), default='cfg')
    parser.add_argument('--num_edge_type', type=int, default=3)
    parser.add_argument('--embedding_dims', type=int, default=128)
    
    parser.add_argument('--seq_model', type=str, default='transformer', choices=('transformer'))
    parser.add_argument('--seq_hidden_dims', type=int, default=128)
    parser.add_argument('--seq_layers', type=int, default=2)
    
    parser.add_argument('--gnn_name', type=str, default='gatedgcn', choices=('gatedgcn'))
    parser.add_argument('--gnn_hidden_dims', type=int, default=128)
    parser.add_argument('--gnn_layers', type=int, default=5)
    parser.add_argument('--gnn_out_dims', type=int, default=128)
    
    parser.add_argument('--train_batch_size', type=int, default=90)
    parser.add_argument('--batch_k', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=90)
    parser.add_argument('--train_num_each_epoch', type=int, default=300000)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--milestones', type=int, nargs='+', default=None)
    parser.add_argument('--miner_type', type=str, default='ms')
    parser.add_argument('--loss_type', type=str, default='ms')
    parser.add_argument('--use_moco', action='store_true', default=False)
    parser.add_argument('--use_xbm', action='store_true', default=False)
    parser.add_argument('--memory_size', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--early_stopping', type=int, default=10)
    
    parser.add_argument('--precision', type=int, default=32, choices=(16, 32))
    parser.add_argument('--save_name', type=str, default='binkit_2.0_cfg_cg_transformer_gatedgcn_gin_moco')
    args = parser.parse_args()

    # dataset
    if args.data_repr == 'cfg_cg':
        DataClass = CFGCGSimDataset
        follow_batch = ['x_cfg', 'x_cg']
    else: raise Exception('Not implemented.')
    
    print(f'using data path: {args.data_dir}')
    train_set = DataClass(root=args.data_dir, mode='train')
    val_set = DataClass(root=args.data_dir, mode='val')
    print(f'Train size: {len(train_set)}. Val size: {len(val_set)}.')
    
    # dataloader
    print('Initialing dataloaders...')
    tik = time.time()
    train_sampler = samplers.MPerClassSampler(
        labels=train_set.data.gid, m=args.batch_k, batch_size=args.train_batch_size,
        length_before_new_iter=args.train_num_each_epoch)
    val_sampler = samplers.FixedSetOfTriplets(
        labels=val_set.data.gid, num_triplets=len(val_set)//3)
    train_loader = pygDataLoader(train_set, args.train_batch_size, sampler=train_sampler, num_workers=args.num_workers, follow_batch=follow_batch)
    val_loader = pygDataLoader(val_set, args.val_batch_size, sampler=val_sampler, num_workers=args.num_workers, follow_batch=follow_batch)
    print(f'Dataloaders init finished: {time.time()-tik}s.')
    
    # model
    model = Transformer_GNN_CG_GNN(
        num_edge_type=args.num_edge_type,
        embedding_dims=args.embedding_dims,
        trans_hidden_dims=args.seq_hidden_dims,
        trans_layers=args.seq_layers,
        gnn_name=args.gnn_name,
        gnn_hidden_dims=args.gnn_hidden_dims,
        gnn_out_dims=args.gnn_out_dims,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        vocab_size=len(train_set.get_vocab()),
        type_vocab_size=len(train_set.get_type_vocab()),
        imp_vocab_size=len(train_set.get_imp_vocab()),
        lr=args.learning_rate,
        milestones=args.milestones,
        miner_type=args.miner_type,
        loss_type=args.loss_type
    )
    
    if args.use_moco:
        model_k = copy.deepcopy(model)
        model = MomentumEncoderWrapper(
            encQ=model, 
            encK=model_k,
            out_dims=args.gnn_out_dims,
            memory_size=args.memory_size,
            momentum=0.999,
            miner_type=args.miner_type,
            loss_type=args.loss_type,
            lr=args.learning_rate,
            milestones=args.milestones
        )

    # callbacks
    ckpt_callback = ModelCheckpoint(
        monitor='val/map', mode='max', save_weights_only=True, 
        filename='epoch{epoch}-step{step}-map{val/map:.4f}-mrr{val/mrr:.4f}',
        auto_insert_metric_name=False)
    early_stopping_callback = EarlyStopping(monitor='val/map', mode='max', patience=args.early_stopping)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir='results/dml', name=args.save_name)

    # trainer
    if args.precision == 16:
        args.precision = '16-mixed'
    elif args.precision == 32:
        args.precision = '32-true'
        
    trainer = Trainer(
        max_epochs=args.num_epochs, 
        accelerator='gpu', devices=[0],
        log_every_n_steps=100, logger=logger,
        callbacks=[ckpt_callback, lr_monitor, early_stopping_callback],
        precision=args.precision)
    
    # save config
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)

    # train
    trainer.fit(model, train_loader, val_loader)

    # save validation result
    val_result = trainer.test(model, val_loader, ckpt_path='best')[0]
    with open(os.path.join(logger.log_dir, 'best_val_result.json'), 'w') as f:
        json.dump(val_result, f, indent=1)


if __name__ == '__main__':
    main()
    
    