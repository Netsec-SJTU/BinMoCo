# @Time: 2022.12.26 20:39
# @Author: Bolun Wu

import argparse
import json
import os
import sqlite3
from collections import Counter

import tqdm
from constants import max_block_token_len, max_num_block
from dataset.normalizer import InsNormalizer
from utils import counter_to_dict


def build_vocab(db_dir):
    os.makedirs(os.path.join(db_dir, 'vocab'), exist_ok=True)
    
    rowids = []
    rowid_filepath = os.path.join(db_dir, 'cleaned_rowids.txt')
    with open(rowid_filepath, 'r') as f:
        for line in f: rowids.append(int(line.strip()))
    
    train_projects = []
    split_filepath = os.path.join(db_dir, 'split.json')
    with open(split_filepath, 'r') as f:
        split = json.load(f)
        train_projects = split['train']

    db_filepath = os.path.join(db_dir, 'db.sqlite')
    conn = sqlite3.connect(db_filepath)
    c = conn.cursor()

    normalizer = InsNormalizer()
    # num_blocks, ins_len, block_len, block_token_len = [], [], [], []
    token_counter, token_type_counter = Counter(), Counter()
    
    for rowid in tqdm.tqdm(rowids, ncols=128):
        query = '''select project, graph from functions where rowid is {}'''.format(rowid)
        c.execute(query); res = c.fetchone()
        project, graph = res
        if project not in train_projects: continue
        
        graph = json.loads(graph)
        # num_blocks.append(len(graph['blocks']))
        
        b_addrs = sorted(graph['blocks'].keys())[:max_num_block]
        
        for b_addr in b_addrs:
            block = graph['blocks'][b_addr]
            # block_len.append(len(block))
            num_tokens = 0
            for ins_record in block:
                tokens, tokens_type, _ = normalizer.parse(ins_record)
                for token, _type in zip(tokens, tokens_type):
                    token_counter[token] += 1
                    token_type_counter[_type] += 1
                    num_tokens += 1
                    if num_tokens == max_block_token_len: break
                if num_tokens == max_block_token_len: break

                # ins_len.append(len(tokens))
                # num_tokens += len(tokens)
            # block_token_len.append(num_tokens)

    conn.close()
    
    token_counter = counter_to_dict(token_counter)
    token_type_counter = counter_to_dict(token_type_counter)

    vocab = ['<PAD>', '<UNK>', '<CLS>'] + list(token_counter.keys())
    vocab = {v: i for i, v in enumerate(vocab)}
    with open(os.path.join(db_dir, 'vocab', 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=1)

    vocab = ['<PAD>', '<UNK>', '<CLS>'] + list(token_type_counter.keys())
    vocab = {v: i for i, v in enumerate(vocab)}
    with open(os.path.join(db_dir, 'vocab', 'type.json'), 'w') as f:
        json.dump(vocab, f, indent=1)

    # num_blocks.sort()
    # ins_len.sort()
    # block_len.sort()
    # block_token_len.sort()
    
    # meta = {'num_blocks': {}, 'ins_len': {}, 'block_len': {}, 'block_token_len': {}}
    # for ratio in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9]:
    #     meta['num_blocks'][ratio] = num_blocks[int(ratio*len(num_blocks))]
    #     meta['ins_len'][ratio] = ins_len[int(ratio*len(ins_len))]
    #     meta['block_len'][ratio] = block_len[int(ratio*len(block_len))]
    #     meta['block_token_len'][ratio] = block_token_len[int(ratio*len(block_token_len))]
        
    # with open(os.path.join(db_dir, 'meta.json'), 'w') as f:
    #     json.dump(meta, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=str, help='database directory')
    args = parser.parse_args()
    
    assert os.path.exists(args.db_dir), 'database directory does not exist'
    
    build_vocab(args.db_dir)
    
    
if __name__ == '__main__':
    main()
    
    