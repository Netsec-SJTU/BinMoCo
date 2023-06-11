# @Time: 2023.5.25 15:28
# @Author: Bolun Wu

import json
import os

import torch
from constants import vul_funcnames
from dataset.pyg_data import CFG_CG_Data, CFGCGSimDataset


def is_vul_funcname(funcname):
    for vul_funcname in vul_funcnames['netgear'] + vul_funcnames['tp-link']:
        if f'sym.{vul_funcname}' == funcname:
            return True
    return False


class CFGCGSimDatasetVul(CFGCGSimDataset):
    def __init__(self, root, vocab_dataset_name, vul_only=False, mode='test'):
        # mode is useless
        self.vul_only = vul_only
        self.vocab_dataset_name = vocab_dataset_name
        self.vocab_dir = os.path.join('database', vocab_dataset_name, 'vocab')
        
        func_cfgs_path = os.path.join(root, 'func_cfgs.json')
        cg_path = os.path.join(root, 'cg.json')
        
        with open(func_cfgs_path, 'r') as f: self.func_cfgs = json.load(f)
        with open(cg_path, 'r') as f: self.cg = self._convert_to_graph_dict(json.load(f))
            
        super(CFGCGSimDatasetVul, self).__init__(root, mode, with_meta=False)
    
    @property
    def processed_dir(self):
        if self.vul_only: prompt = 'vul'
        else: prompt = 'all'
        return os.path.join(self.root, f'{prompt}_{self.vocab_dataset_name}_cfg_cg_pyg')

    @property
    def funcname_list(self):
        if self.vul_only:
            return [funcname for funcname in self.func_cfgs if is_vul_funcname(funcname)]
        else:
            return list(self.func_cfgs.keys())

    @property
    def imp_vocab_path(self):
        return os.path.join(self.vocab_dir, 'imp_vocab.json')
    
    @property
    def vocab_path(self):
        return os.path.join(self.vocab_dir, 'vocab.json')
    
    @property
    def vocab_type_path(self):
        return os.path.join(self.vocab_dir, 'type.json')

    def process(self):
        import tqdm
        self.vocab = self.get_vocab()
        self.type_vocab = self.get_type_vocab()
        self.imp_vocab = self.get_imp_vocab()
        
        graph_list = []
        for funcname in tqdm.tqdm(self.funcname_list):
            cfg = self.func_cfgs[funcname]
            sub_cg = self._get_subgraph(self.cg, funcname)
            
            (xs, xs_type, padding_masks, edges, edges_type, 
            x_cg, edge_index_cg) = self._extract_graph(cfg, sub_cg)
            
            graph = CFG_CG_Data(
                # cfg
                x_cfg=torch.tensor(xs, dtype=torch.int16),
                x_type=torch.tensor(xs_type, dtype=torch.int16),
                padding_masks=torch.tensor(padding_masks, dtype=torch.int8),
                edge_index_cfg=torch.tensor(edges, dtype=torch.int32).t(),
                edge_type=torch.tensor(edges_type, dtype=torch.int16),
                
                # cg
                x_cg=torch.tensor(x_cg, dtype=torch.int16),
                edge_index_cg=torch.tensor(edge_index_cg, dtype=torch.int32).t(),                    
            )
            
            graph_list.append(graph)
        torch.save(self.collate(graph_list), self.processed_paths[0])


