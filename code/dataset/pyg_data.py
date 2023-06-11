# @Time: 2022.12.15 15:32
# @Author: Bolun Wu

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from collections import deque

import torch
from constants import (compile_config, max_block_token_len, max_cg_in_degree,
                       max_cg_out_degree, max_num_block)
from dataset.normalizer import InsNormalizer
from torch_geometric.data import Data, InMemoryDataset


class BinFuncSimDataset(InMemoryDataset):
    def __init__(self, root, mode, with_meta=True):
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        
        if with_meta:
            
            if 'binkit_2.0' in root:
                configs = compile_config['binkit_2.0']
            elif 'binkit' in root:
                configs = compile_config['binkit']
            elif 'sec22' in root:
                configs = compile_config['sec22']
            else:
                raise NotImplementedError
            
            self.arch_dict = {k: i for i, k in enumerate(configs['arch'])}
            self.bit_dict = {k: i for i, k in enumerate(configs['bit'])}
            self.comp_dict = {k: i for i, k in enumerate(configs['compiler'])}
            self.opti_dict = {k: i for i, k in enumerate(configs['opti'])}
        
        super(BinFuncSimDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        raise NotImplementedError
    
    @property
    def processed_file_names(self):
        return [f'{self.mode}.pt']
    
    @property
    def db_cg_path(self):
        return os.path.join(self.root, 'db_cg.sqlite')
    
    @property
    def db_path(self):
        return os.path.join(self.root, 'db.sqlite')
    
    @property
    def cleaned_rowids_path(self):
        return os.path.join(self.root, 'cleaned_rowids.txt')
    
    @property
    def group_path(self):
        return os.path.join(self.root, 'group.csv')

    @property
    def data_split_path(self):
        return os.path.join(self.root, 'split.json')
    
    @property
    def vocab_path(self):
        return os.path.join(self.root, 'vocab', 'vocab.json')
    
    @property
    def vocab_type_path(self):
        return os.path.join(self.root, 'vocab', 'type.json')
    
    def get_data_split(self):
        with open(self.data_split_path, 'r') as f:
            data_split = json.load(f) # dict({'train': ... 'val': ... 'test': ...})
        return data_split
    
    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def get_type_vocab(self):
        with open(self.vocab_type_path, 'r') as f:
            type_vocab = json.load(f)
        return type_vocab
    
    def process(self):
        raise NotImplementedError


class CFG_CG_Data(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_cfg':
            return self.x_cfg.size(0)
        if key == 'edge_index_cg':
            return self.x_cg.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class CFGCGSimDataset(BinFuncSimDataset):
    def __init__(self, root, mode, cg_hop=2, **kw):
        """
        cg_hop (int): consider neighbors within cg_hop hops in call graph
        """
        self.cg_hop = cg_hop
        print(f'{mode}: using control flow graph and call graph.')
        super(CFGCGSimDataset, self).__init__(root, mode, **kw)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'cfg_cg_pyg')
    
    @property
    def imp_vocab_path(self):
        return os.path.join(self.root, 'vocab', 'imp_vocab.json')
        
    def get_imp_vocab(self):
        with open(self.imp_vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def _convert_to_graph_dict(self, cg):
        """
        Args:
            cg (list of dict): call graph
        Returns:
            graph (dict): {func_name: {'func_name': str, 'children': list, 'parents': list}}
        """
        graph = dict()
        for node_info in cg:
            func_name = node_info['name']
            callees = node_info['imports']
            if func_name not in graph:
                graph[func_name] = {'func_name': func_name, 'children': [], 'parents': []}
            for callee in callees:
                if callee not in graph:
                    graph[callee] = {'func_name': callee, 'children': [], 'parents': []}
            graph[func_name]['children'] += callees
            for callee in callees:
                graph[callee]['parents'].append(func_name)
        return graph
    
    def _get_subgraph(self, cg, func_name):
        """get `self.cp_hop` sub-graph centered at `func_name`
        two circumstances: no func_name in cg, or func_name in cg
        """
        sub_cg = dict()
        if func_name not in cg:
            sub_cg[func_name] = {'children': [], 'parents': []}
            return sub_cg
        cur_node = cg[func_name]
        
        queue = deque()
        queue.appendleft(cur_node)
        for i in range(self.cg_hop):
            size = len(queue)
            for _ in range(size):
                cur_node = queue.pop()
                # insert node into sub_cg
                if cur_node['func_name'] not in sub_cg: 
                    sub_cg[cur_node['func_name']] = {'children': [], 'parents': []}
                for n in cur_node['children']:
                    if n not in sub_cg:
                        sub_cg[n] = {'children': [], 'parents': []}
                    # insert edge into sub_cg
                    sub_cg[cur_node['func_name']]['children'].append(n)
                    sub_cg[n]['parents'].append(cur_node['func_name'])
                if i == self.cg_hop - 1: continue
                for n in cur_node['children']:
                    queue.appendleft(cg[n])

        queue.clear()
        
        cur_node = cg[func_name]
        queue.appendleft(cur_node)
        for i in range(self.cg_hop):
            size = len(queue)
            for _ in range(size):
                cur_node = queue.pop()
                if cur_node['func_name'] not in sub_cg:
                    sub_cg[cur_node['func_name']] = {'children': [], 'parents': []}
                for n in cur_node['parents']:
                    if n not in sub_cg:
                        sub_cg[n] = {'children': [], 'parents': []}
                    # insert edge into sub_cg
                    sub_cg[n]['children'].append(cur_node['func_name'])
                    sub_cg[cur_node['func_name']]['parents'].append(n)
                if i == self.cg_hop - 1: continue
                for n in cur_node['parents']:
                    queue.appendleft(cg[n])

        return sub_cg
        
    def _extract_graph(self, graph, sub_cg):
        normalizer = InsNormalizer()
        if self.mode == 'test':
            b_addrs = sorted(graph['blocks'].keys())
        else:
            b_addrs = sorted(graph['blocks'].keys())[:max_num_block]
        b_addr_to_id = dict()
        
        ## cfg nodes: x, x_type, padding_masks
        xs, xs_type, padding_masks = [], [], []
        for b_addr in b_addrs:
            
            block = graph['blocks'][b_addr]
            x, x_type = [], []
            for ins_record in block:
                tokens, tokens_type, _ = normalizer.parse(ins_record)
                for token, _type in zip(tokens, tokens_type):
                    if token in self.vocab: token_id = self.vocab[token]
                    else: token_id = self.vocab['<UNK>']
                    if _type in self.type_vocab: type_id = self.type_vocab[_type]
                    else: type_id = self.type_vocab['<UNK>']
                    x.append(token_id); x_type.append(type_id)
                    if len(x) == max_block_token_len: break
                if len(x) == max_block_token_len: break
            
            # ! special case: fid 706514, which causes Transformer to produce nan
            if len(x) == 0: continue
            
            # add <CLS> token
            x = [self.vocab['<CLS>']] + x
            x_type = [self.type_vocab['<CLS>']] + x_type
            
            # padding: padding mask is different from attention mask in HuggingFace
            # padding_mask = ~attention_mask
            padding_mask = [0 for _ in range(len(x))]
            for _ in range(max_block_token_len + 1 - len(x)):
                x.append(self.vocab['<PAD>'])
                x_type.append(self.type_vocab['<PAD>'])
                padding_mask.append(1)
            
            b_addr_to_id[b_addr] = len(b_addr_to_id)
            xs.append(x)
            xs_type.append(x_type)
            padding_masks.append(padding_mask)

        ## cfg edges: edge_index, edge_type
        edges, edges_type = [], []
        for e in graph['cfg']:
            src, dst, e_type = e # true jump (1), false jump (0), unconditional (2)
            assert e_type in (0, 1, 2)
            src, dst = str(src), str(dst)
            if (src not in b_addr_to_id) or (dst not in b_addr_to_id): continue
        
            edges.append([b_addr_to_id[src], b_addr_to_id[dst]])
            edges_type.append(e_type)
        
        # call graph
        ## cg nodes: x
        x_cg = []
        func_name_to_id = dict()
        for node in sub_cg:
            # node is actually func_name
            if node.startswith('sym.imp.'):
                name_id = self.imp_vocab.get(node, self.imp_vocab['<UNK>'])
            else:
                name_id = self.imp_vocab['<INTERNAL>']
                
            in_degree = len(sub_cg[node]['parents'])
            out_degree = len(sub_cg[node]['children'])
            
            in_degree = min(in_degree, max_cg_in_degree)
            out_degree = min(out_degree, max_cg_out_degree)
            
            x_cg.append([name_id, in_degree, out_degree])
            func_name_to_id[node] = len(func_name_to_id)

        ## cg edges: edge_index
        edge_index_cg = []
        for node in sub_cg:
            for dst in sub_cg[node]['children']:
                edge_index_cg.append([func_name_to_id[node], func_name_to_id[dst]])

        return xs, xs_type, padding_masks, edges, edges_type, x_cg, edge_index_cg

    def extract_graph(self, fid, gid):
        # control flow graph
        cur = self.conn.cursor()
        cur.execute(
            '''select 
            file_name, function_name, architecture, bits, 
            compiler, compiler_version, optimization, graph
            from functions where rowid is {}
            '''.format(fid)
        ) # select by rowid
        res = cur.fetchone()
        
        file_name, func_name, arch, bits, compiler, comp_ver, opti, graph = res

        graph = json.loads(graph)
        
        # call graph
        ## parse cg into networkx format and store to memory
        if file_name not in self.filename_to_nx_cg:
            cur_cg = self.conn_cg.cursor()
            cur_cg.execute(
                '''select 
                file_name, cg from binary_cg 
                where file_name="{}"
                '''.format(file_name)
            )
            res = cur_cg.fetchone() # file_name, cg
            assert res[0] == file_name
            cg = json.loads(res[1])
            self.filename_to_nx_cg[file_name] = self._convert_to_graph_dict(cg)
        
        cg = self.filename_to_nx_cg[file_name]
        sub_cg = self._get_subgraph(cg, func_name)

        (xs, xs_type, padding_masks, edges, edges_type, 
         x_cg, edge_index_cg) = self._extract_graph(graph, sub_cg)
        
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
            
            # meta / label
            fid=torch.tensor(fid, dtype=torch.int32),
            gid=torch.tensor(gid, dtype=torch.int32),
            comp=torch.tensor(self.comp_dict[f'{compiler}-{comp_ver}'], dtype=torch.int8),
            opti=torch.tensor(self.opti_dict[opti], dtype=torch.int8),
            arch=torch.tensor(self.arch_dict[arch], dtype=torch.int8),
            bit=torch.tensor(self.bit_dict[bits], dtype=torch.int8)                     
        )

        return graph
        
    def process(self):
        import sqlite3

        import pandas as pd
        import tqdm

        # get project names according to mode (train val test)
        data_split_dict = self.get_data_split()
        projects = data_split_dict[self.mode]
        
        # get samples to be parsed [(fid, gid), ...]
        fid_gid_list = []
        group_df = pd.read_csv(self.group_path)
        for i, row in group_df.iterrows():
            project = row['project']
            if project not in projects: continue
            
            rowids = list(map(lambda x: int(x), row['rowids'].split()))
            for rowid in rowids: fid_gid_list.append([rowid, i])
        
        # process
        self.vocab = self.get_vocab()
        self.type_vocab = self.get_type_vocab()
        self.imp_vocab = self.get_imp_vocab()
        self.filename_to_nx_cg = {}
        
        ## open corresponding dbs
        self.conn = sqlite3.connect(self.db_path)
        self.conn_cg = sqlite3.connect(self.db_cg_path)
        
        graph_list = []
        for fid, gid in tqdm.tqdm(fid_gid_list, ncols=80):
            graph_list.append(self.extract_graph(fid=fid, gid=gid))
        
        ## close dbs
        self.conn_cg.close()
        self.conn.close()
        
        torch.save(self.collate(graph_list), self.processed_paths[0])
        

def main():
    # dataset_name = 'binkit_2.0'
    dataset_name = 'sec22'
    
    train_set = CFGCGSimDataset(root=f'database/{dataset_name}', mode='train')
    val_set = CFGCGSimDataset(root=f'database/{dataset_name}', mode='val')
    test_set = CFGCGSimDataset(root=f'database/{dataset_name}', mode='test')
    print(f'Train size: {len(train_set)}. Val size: {len(val_set)}. Test size: {len(test_set)}.')


if __name__ == '__main__':
    main()

