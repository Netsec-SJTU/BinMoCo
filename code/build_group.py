# @Time: 2023.5.4 16:17
# @Author: Bolun Wu

import argparse
import os
import time

import connectorx as cx
import pandas as pd
import tqdm
from constants import min_num_block, min_variant_num


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=str, help='database directory')
    args = parser.parse_args()
    
    db_dir = args.db_dir
    db_path = os.path.join(db_dir, 'db.sqlite')
    
    query = '''
        SELECT 
        rowid, file_name, project, binary_name,
        architecture, bits, compiler, compiler_version, optimization, 
        function_name, num_block 
        FROM
        functions
    '''

    print(f'loading database {db_path} into pandas...')
    tik = time.time()
    df = cx.read_sql(f'sqlite://{db_path}', query)
    print(f'done loading database into pandas: {time.time()-tik:.2f}s')
    
    # * filter functions
    # 1. remove blocks less than 5
    orig_size = len(df)
    df = df[df['num_block'] >= min_num_block]
    cur_size = len(df)
    print(f'remove blocks < {min_num_block}: {orig_size} -> {cur_size} {cur_size/orig_size*100:.2f}%.')
    
    # NOTE
    # functions with 'entry' and 'fcn.' and '.imp.' have been removed when building db
    # so we don't need to remove them here
    # 2.(1) remove functions with 'std::'
    orig_size = len(df)
    df = df[df['function_name'].map(lambda x: 'std::' not in str(x))]

    # 2.(2) remove functions in z3 with 'sym.__'
    df = df[(df['project']!='z3') | (df['function_name'].map(lambda x: 'sym.__' not in str(x)))]
    cur_size = len(df)
    print(f'remove functions with "std::" and z3 functions with "sym.__": {orig_size} -> {cur_size} {cur_size/orig_size*100:.2f}%.')

    # 3. remove duplicated functions shared in the same project
    checked_columns = [
        'project', 'architecture', 'bits',
        'compiler', 'compiler_version', 'optimization',
        'function_name'
    ]
    orig_size = len(df)
    df.drop_duplicates(subset=checked_columns, keep='first', inplace=True)
    cur_size = len(df)
    print(f'remove duplicated functions shared in the same project: {orig_size} -> {cur_size} {cur_size/orig_size*100:.2f}%.')
    
    # ? remove duplicated functions shared by different projects

    # 4. remove functions with less than 5 variants
    df_grouped = df.groupby(by=['function_name', 'project'], as_index=False).agg(
        count=pd.NamedAgg(column='file_name', aggfunc=len)
    )
    df_grouped.sort_values(by='count', ascending=False, inplace=True)
    df_grouped['signature'] = df_grouped['function_name'] + df_grouped['project']
    sig_to_be_discard = df_grouped[df_grouped['count'] < min_variant_num]['signature'].tolist()
    
    orig_size = len(df)
    df['signature'] = df['function_name'] + df['project']
    df = df[~df['signature'].isin(sig_to_be_discard)]
    cur_size = len(df)
    print(f'remove functions with < {min_variant_num} variants: {orig_size} -> {cur_size} {cur_size/orig_size*100:.2f}%.')

    # * generate function groups    
    groups = dict()
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), ncols=128, desc='gen groups:'):
        sig = '\t'.join([row['project'], row['function_name']])
        if sig not in groups: groups[sig] = []
        groups[sig].append(row['rowid'])
    
    # * save groups and all rowids
    cleaned_rowids = []
    with open(os.path.join(db_dir, 'group.csv'), 'w') as f:
        f.write('project,function_name,rowids\n')
        for k, v in groups.items():
            cleaned_rowids.extend(v)
            project, func_name = k.split('\t')
            v = list(map(lambda x: str(x), v))
            v = ' '.join(v)
            f.write(f'{project},{func_name},{v}\n')
    
    cleaned_rowids.sort()
    with open(os.path.join(db_dir, 'cleaned_rowids.txt'), 'w') as f:
        for rowid in cleaned_rowids: f.write(f'{rowid}\n')
    
    print(f'Finally remain {len(cleaned_rowids)} functions.')
    print(f'Number of distinct function: {len(groups)}.')

    
if __name__ == '__main__':
    main()
    
    