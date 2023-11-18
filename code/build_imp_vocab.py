import argparse
import json
import os
import sqlite3

import tqdm


def build_imp_vocab(db_dir):
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

    conn = sqlite3.connect(os.path.join(db_dir, 'db.sqlite'))
    c = conn.cursor()
    train_filenames = set()
    for rowid in tqdm.tqdm(rowids, ncols=80, desc='get train fids'):
        query = '''select project, file_name from functions where rowid is {}'''.format(rowid)
        c.execute(query); res = c.fetchone()
        project, filename = res
        if project not in train_projects: continue
        train_filenames.add(filename)
    conn.close()
    
    train_filenames = list(train_filenames)
    conn = sqlite3.connect(os.path.join(db_dir, 'db_cg.sqlite'))
    c = conn.cursor()
    
    imports = set()
    for train_filename in tqdm.tqdm(train_filenames, ncols=80, desc='get import funcnames'):
        query = '''select file_name, cg from binary_cg where file_name is "{}"'''.format(train_filename)
        c.execute(query); res = c.fetchone()
        filename, cg = res
        cg = json.loads(cg)
        for node_info in cg:
            callees = node_info['imports']
            for callee in callees:
                if callee.startswith('sym.imp.'):
                    imports.add(callee)
    conn.close()
    
    imp_vocab = ['<UNK>', '<INTERNAL>'] + sorted(list(imports))
    imp_vocab = {v: i for i, v in enumerate(imp_vocab)}
    with open(os.path.join(db_dir, 'vocab', 'imp_vocab.json'), 'w') as f:
        json.dump(imp_vocab, f, indent=1)
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=str, help='database directory')
    args = parser.parse_args()
    
    build_imp_vocab(args.db_dir)


if __name__ == '__main__':
    main()
    
    