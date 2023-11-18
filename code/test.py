import argparse
import copy
import json
import os
import pickle

import lightning as L
import numpy as np
import torch
import tqdm
from constants import seed
from dataset.pyg_data import CFG_CG_Data, CFGCGSimDataset
from models.bcsd import MomentumEncoderWrapper, Transformer_GNN_CG_GNN
from models.knn import FaissKNN
from models.metrics import RankingMetrics
from torch_geometric.loader import DataLoader


def infer(model, test_set, save_dir, follow_batch):
    # init
    torch.set_grad_enabled(False)
    L.seed_everything(seed)
    
    test_feat_path = os.path.join(save_dir, 'test_feat.pickle')
    if not os.path.exists(test_feat_path):
        fabric = L.Fabric(accelerator='cuda', devices=1)
        fabric.launch()
        
        # model
        model.eval()
        model = fabric.setup_module(model)

        # dataloader
        test_loader = fabric.setup_dataloaders(
            DataLoader(test_set, batch_size=8, shuffle=True, num_workers=12, follow_batch=follow_batch))

        fid_to_feat = {}
        for data in tqdm.tqdm(test_loader, ncols=80):
            feats = model(data)

            fids = data.fid.cpu().numpy().squeeze()
            feats = feats.cpu().numpy().squeeze()
            for fid, feat in zip(fids, feats):
                fid_to_feat[fid] = feat
        
        with open(test_feat_path, 'wb') as f:
            pickle.dump(fid_to_feat, f)

    else:
        with open(test_feat_path, 'rb') as f:
            fid_to_feat = pickle.load(f)
    
    return fid_to_feat


def evaluate_task(test_path, fid_to_feat):
    metricer = RankingMetrics()
    knner = FaissKNN(use_gpu=False)

    count = 0
    mrr, recalls = 0.0, {'@1': 0, '@5': 0, '@10': 0, '@20': 0, '@30': 0, '@40': 0, '@50': 0}
    f = open(test_path, 'r')
    for line in tqdm.tqdm(f, total=10000, ncols=80):
        line = line.strip().split(',')
        line = list(map(lambda x: int(x), line))
        
        query_sample, pool = line[0], line[1:]
        
        query = torch.tensor(fid_to_feat[query_sample]).unsqueeze(0)
        reference = torch.tensor(np.stack([fid_to_feat[x] for x in pool], axis=0))
        
        query_labels = torch.tensor([1])
        reference_labels = torch.tensor([1] + [0 for _ in range(len(pool)-1)])
        _, knn_indexes = knner(query=query, reference=reference, k=reference.size(0))        
        knn_labels = reference_labels[knn_indexes]
        rele_num_per_query = torch.tensor([1])
        
        _mrr = metricer.mean_reciprocal_rank(knn_labels, query_labels).item()
        _recall_at_ks = metricer.recall_at_ks(knn_labels, query_labels, rele_num_per_query)
        
        mrr += _mrr
        for at_k in _recall_at_ks: recalls[at_k] += _recall_at_ks[at_k]
        
        count += 1
        
    f.close()

    mrr /= count
    for k in recalls: recalls[k] /= count
    
    return mrr, recalls


def test_task(test_dir, task, pool_size, fid_to_feat):
    test_path = os.path.join(test_dir, f'test_{task.lower()}_{pool_size}.csv')
    mrr, recalls = evaluate_task(test_path, fid_to_feat)
    ret_info = {
        'task': task,
        'pool_size': pool_size,
        'mrr': mrr,
        'recall@1': recalls['@1'],
        'recall@5': recalls['@5'],
        'recall@10': recalls['@10'],
        'recall@20': recalls['@20'],
        'recall@30': recalls['@30'],
        'recall@40': recalls['@40'],
        'recall@50': recalls['@50'],
    }
    print(f'{task} (poolsize={pool_size}): mrr {mrr:.4f}, recall@1 {recalls["@1"]:.4f}, recall@5 {recalls["@5"]:.4f}, recall@10 {recalls["@10"]:.4f}.')
    return ret_info
    
    
def test(test_tasks_dir, fid_to_feat):
    import pandas as pd

    metric_df = pd.DataFrame(columns=[
        'task', 'pool_size', 'mrr', 'recall@1', 'recall@5', 
        'recall@10', 'recall@20', 'recall@30', 'recall@40', 'recall@50'
    ])
    
    for task_name in ['XO', 'XC', 'XA', 'XM']:
        for pool_size in [2, 10, 100, 500, 1000, 5000, 10000]:
            ret_info = test_task(test_tasks_dir, task_name, pool_size, fid_to_feat)
            metric_df = pd.concat([metric_df, pd.DataFrame([ret_info])])

    return metric_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str) # save_dir = 'results/binkit_lstm_gnn/version_0'
    args = parser.parse_args()
    
    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    if config['data_repr'] == 'cfg_cg': 
        DataClass = CFGCGSimDataset
        follow_batch = ['x_cfg', 'x_cg']
    else: raise Exception('Not implemented data representation.')
    
    print(f'using data path: {config["data_dir"]}')
    test_set = DataClass(root=config['data_dir'], mode='test')
    print(f'Test size: {len(test_set)}.')
    
    model = Transformer_GNN_CG_GNN(
        num_edge_type=config['num_edge_type'],
        embedding_dims=config['embedding_dims'],
        trans_hidden_dims=config['seq_hidden_dims'],
        trans_layers=config['seq_layers'],
        gnn_name=config['gnn_name'],
        gnn_hidden_dims=config['gnn_hidden_dims'],
        gnn_out_dims=config['gnn_out_dims'],
        gnn_layers=config['gnn_layers'],
        vocab_size=len(test_set.get_vocab()),
        type_vocab_size=len(test_set.get_type_vocab()),
        imp_vocab_size=len(test_set.get_imp_vocab())
    )
    
    if config['use_moco']:
        model_k = copy.deepcopy(model)
        model = MomentumEncoderWrapper(
            encQ=model, encK=model_k,
            out_dims=config['gnn_out_dims'],
            memory_size=config['memory_size']
        )

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    model_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    
    fid_to_feat = infer(model, test_set, args.save_dir, follow_batch)
    
    test_tasks_dir = os.path.join(config['data_dir'], 'test_tasks')
    metric_df = test(test_tasks_dir, fid_to_feat)
    metric_df.to_csv(os.path.join(args.save_dir, 'test_results.csv'), index=False)
    
    
if __name__ == '__main__':
    main()
    
    