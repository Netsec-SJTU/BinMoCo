# @Time: 2023.5.25 10:49
# @Author: Bolun Wu

import argparse
import copy
import json
import os

import lightning as L
import pandas as pd
import torch
import tqdm
from constants import cve_dict, seed, vul_funcnames
from dataset.vul_pyg_data import CFGCGSimDatasetVul
from models.bcsd import MomentumEncoderWrapper, Transformer_GNN_CG_GNN
from models.knn import FaissKNN
from models.metrics import RankingMetrics
from torch_geometric.loader import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()
    
    # config
    with open(os.path.join(args.save_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    origin_model_dataset_name = config['data_dir'].split('/')[-1]
    
    # perpare dataset
    name_to_dataclass = {}
    data_dir = 'database/vul_libcrypto.so.1.0.0'
    for dirname in os.listdir(data_dir):
        if 'openssl' in dirname: # query
            name = f'query_{dirname.split("_")[-1]}'
            name_to_dataclass[name] = CFGCGSimDatasetVul(
                root=os.path.join(data_dir, dirname), 
                vocab_dataset_name=origin_model_dataset_name,
                vul_only=True)
        else: # target pools
            name = f'target_{dirname.split("_")[1]}'.lower()
            name_to_dataclass[name] = CFGCGSimDatasetVul(
                root=os.path.join(data_dir, dirname), 
                vocab_dataset_name=origin_model_dataset_name,
                vul_only=False)
    
    vocab_size = len(name_to_dataclass['query_x86'].get_vocab())
    type_vocab_size = len(name_to_dataclass['query_x86'].get_type_vocab())
    imp_vocab_size = len(name_to_dataclass['query_x86'].get_imp_vocab())
    
    model = Transformer_GNN_CG_GNN(
        num_edge_type=config['num_edge_type'],
        embedding_dims=config['embedding_dims'],
        trans_hidden_dims=config['seq_hidden_dims'],
        trans_layers=config['seq_layers'],
        gnn_name=config['gnn_name'],
        gnn_hidden_dims=config['gnn_hidden_dims'],
        gnn_out_dims=config['gnn_out_dims'],
        gnn_layers=config['gnn_layers'],
        vocab_size=vocab_size,
        type_vocab_size=type_vocab_size,
        imp_vocab_size=imp_vocab_size
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
    model.eval()

    fabric = L.Fabric(accelerator='cuda', devices=1)
    fabric.launch()
    model = fabric.setup_module(model)
    
    # inference for queries and pools
    torch.set_grad_enabled(False)
    L.seed_everything(seed)
    name_to_funcname_to_repr = {}
    for name, dataclass in name_to_dataclass.items():       
        name_to_funcname_to_repr[name] = {}
        data_loader = fabric.setup_dataloaders(
            DataLoader(dataclass, batch_size=8, shuffle=False, follow_batch=['x_cfg', 'x_cg']))
        
        reprs = []
        for data in tqdm.tqdm(data_loader, ncols=80):
            feats = model(data)
            feats = feats.cpu().squeeze().tolist()
            reprs.extend(feats)
        
        for i, funcname in enumerate(dataclass.funcname_list):
            name_to_funcname_to_repr[name][funcname] = reprs[i]
    
    # result df
    # query_name, query_funcname, cve_id, target_name, rank, mrr, recall@{1,5,10,20,30,40,50}
    result_df = pd.DataFrame(columns=[
        'query_name', 'query_funcname', 'cve_id', 'target_name', 'rank',
        'mrr', 'recall@1', 'recall@5', 'recall@10', 'recall@20', 
        'recall@30', 'recall@40', 'recall@50'
    ])
    
    for device_name, query_funcnames in vul_funcnames.items():
        for query_funcname in query_funcnames:
            for query_name in ['query_x86', 'query_x64', 'query_arm32', 'query_mips32']:
                query = torch.tensor(name_to_funcname_to_repr[query_name][f'sym.{query_funcname}']).unsqueeze(0)
                
                _name = f'target_{device_name}'
                reference = torch.stack(
                    [torch.tensor(name_to_funcname_to_repr[_name][x]) 
                     for x in name_to_funcname_to_repr[_name]], 
                    axis=0
                )
                
                query_labels = torch.tensor([1])
                reference_labels = []
                for funcname in name_to_funcname_to_repr[_name]:
                    if funcname == f'sym.{query_funcname}':
                        reference_labels.append(1)
                    else:
                        reference_labels.append(0)
                reference_labels = torch.tensor(reference_labels)
                
                _, knn_indexes = FaissKNN(use_gpu=False)(query=query, reference=reference, k=reference.size(0))
                knn_labels = reference_labels[knn_indexes]
                rele_num_per_query = torch.tensor([1])
                
                # metrics
                rank = (knn_labels==1).nonzero()[0][1].item() + 1
                mrr = RankingMetrics().mean_reciprocal_rank(knn_labels, query_labels).item()
                recall_at_ks = RankingMetrics().recall_at_ks(knn_labels, query_labels, rele_num_per_query)

                insert_info = {
                    'query_name': query_name.split('_')[1],
                    'query_funcname': query_funcname,
                    'cve_id': cve_dict[query_funcname],
                    'target_name': device_name,
                    'rank': rank,
                    'mrr': mrr,
                    'recall@1': recall_at_ks['@1'],
                    'recall@5': recall_at_ks['@5'],
                    'recall@10': recall_at_ks['@10'],
                    'recall@20': recall_at_ks['@20'],
                    'recall@30': recall_at_ks['@30'],
                    'recall@40': recall_at_ks['@40'],
                    'recall@50': recall_at_ks['@50']
                }
                
                result_df = pd.concat([result_df, pd.DataFrame([insert_info])])
    
    print(result_df)
    result_df.to_csv(os.path.join(args.save_dir, 'vul_detect_result.csv'), index=False)


if __name__ == '__main__':
    main()

