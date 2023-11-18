import argparse
import json
import os
import sqlite3

import tqdm
from constants import seed


def write_list_to_file(f_handle, list_to_write):
    list_to_write = [str(x) for x in list_to_write]
    f_handle.write(','.join(list_to_write) + '\n')


def build_search_tasks(db_dir):
    """build 4 different search test tasks and return rowids
    each task has different pool size settings (2, 10, 100, 500, 1000, 5000, 10000)
    pool size means the model has to select the only one similar sample from total `pool size` samples
    each task contains 10000 queries
    
    XO: diff optimization
    XC: diff compiler + diff optimization
    XA: diff architecture
    XM: diff architecture + diff compiler + diff optimization
    
    Args:
        db_dir (str): database directory
    """
    import random

    import pandas as pd
    random.seed(seed)
    
    test_dir = os.path.join(db_dir, 'test_tasks')
    os.makedirs(test_dir, exist_ok=True)
    
    # * save information of testset as csv
    test_info_path = os.path.join(test_dir, 'test_info.csv')
    if not os.path.exists(test_info_path): 
        # get test rowids
        with open(os.path.join(db_dir, 'split.json'), 'r') as f:
            split = json.load(f)
        test_projs = split['test']
        
        # get groups
        group_df = pd.read_csv(os.path.join(db_dir, 'group.csv'))
        test_rowids = []
        for _, row in group_df.iterrows():
            project = row['project']
            if project not in test_projs: continue
            
            rowids = list(map(lambda x: int(x), row['rowids'].split()))
            for rowid in rowids: test_rowids.append(rowid)
        
        # get meta dataframe from database for test rowids
        query = '''
            SELECT
            rowid, project, function_name, source_line,
            architecture, bits, compiler, compiler_version, optimization
            FROM functions 
            WHERE rowid is {}
        '''
        
        f = open(test_info_path, 'w')
        f.write('rowid,source,arch,comp,opti\n')

        conn = sqlite3.connect(os.path.join(db_dir, 'db.sqlite'))
        c = conn.cursor()
        for _, rowid in enumerate(tqdm.tqdm(test_rowids)):
            c.execute(query.format(rowid))
            res = c.fetchone()
            rowid, project, function_name, source_line, architecture, bits, compiler, compiler_version, optimization = res
            source = '|'.join([project, function_name, source_line])
            arch = architecture + '_' + bits
            comp = compiler + '_' + compiler_version
            opti = optimization
            f.write(f'{rowid},{source},{arch},{comp},{opti}\n')
        conn.close()        
        f.close()
        
    df = pd.read_csv(test_info_path)
    
    # * XO (diff optimization)
    if not all(os.path.exists(os.path.join(test_dir, f'test_xo_{i}.csv')) for i in [2, 10, 100, 500, 1000, 5000, 10000]):
        count, pbar = 0, tqdm.tqdm(desc='XO testset: ', total=10000)    
        target_idx = random.sample(list(range(len(df))), len(df)//2)
        
        f_2 = open(os.path.join(test_dir, 'test_xo_2.csv'), 'w')
        f_10 = open(os.path.join(test_dir, 'test_xo_10.csv'), 'w')
        f_100 = open(os.path.join(test_dir, 'test_xo_100.csv'), 'w')
        f_500 = open(os.path.join(test_dir, 'test_xo_500.csv'), 'w')
        f_1000 = open(os.path.join(test_dir, 'test_xo_1000.csv'), 'w')
        f_5000 = open(os.path.join(test_dir, 'test_xo_5000.csv'), 'w')
        f_10000 = open(os.path.join(test_dir, 'test_xo_10000.csv'), 'w')
        
        for idx in target_idx:
            row = df.iloc[idx]
            rowid, source, arch, comp, opti = row['rowid'], row['source'], row['arch'], row['comp'], row['opti']
            
            tmp_df = df[(df['arch'] == arch) & (df['comp'] == comp) & (df['opti'] != opti)]
            same_source_df = tmp_df[(tmp_df['source'] == source) & (tmp_df['rowid'] != rowid)]
            diff_source_df = tmp_df[tmp_df['source'] != source]
        
            if len(diff_source_df) < 9999: continue
            
            ## get 1 positive sample
            rowids = list(same_source_df['rowid'])
            if len(rowids) == 0: continue
            pos_rowid = random.sample(rowids, 1)[0]
            
            ## get 1 negative sample (poolsize=2)
            rowids = list(diff_source_df['rowid'])
            neg_rowids = random.sample(rowids, 1)
            write_list_to_file(f_2, [rowid, pos_rowid] + neg_rowids)
            
            ## get 9 negative sample (poolsize=10)
            neg_rowids = random.sample(rowids, 9)
            write_list_to_file(f_10, [rowid, pos_rowid] + neg_rowids)
            
            ## get 99 negative samples (poolsize=100)
            neg_rowids = random.sample(rowids, 99)
            write_list_to_file(f_100, [rowid, pos_rowid] + neg_rowids)

            ## get 499 negative samples (poolsize=500)
            neg_rowids = random.sample(rowids, 499)
            write_list_to_file(f_500, [rowid, pos_rowid] + neg_rowids)
            
            ## get 999 negative samples (poolsize=1000)
            neg_rowids = random.sample(rowids, 999)
            write_list_to_file(f_1000, [rowid, pos_rowid] + neg_rowids)
            
            ## get 4999 negative samples (poolsize=5000)
            neg_rowids = random.sample(rowids, 4999)
            write_list_to_file(f_5000, [rowid, pos_rowid] + neg_rowids)
            
            neg_rowids = random.sample(rowids, 9999)
            write_list_to_file(f_10000, [rowid, pos_rowid] + neg_rowids)

            count += 1; pbar.update()
            if count == 10000: break
            
        if count != 10000: raise Exception('XM testset: not enough samples')
        
        f_10000.close(); f_5000.close(); f_1000.close(); f_500.close(); f_100.close(); f_10.close(); f_2.close()
        pbar.close()
    
    # * XC (diff compiler + diff optimization)
    if not all(os.path.exists(os.path.join(test_dir, f'test_xc_{i}.csv')) for i in [2, 10, 100, 500, 1000, 5000, 10000]):
        count, pbar = 0, tqdm.tqdm(desc='XC testset: ', total=10000)    
        target_idx = random.sample(list(range(len(df))), len(df)//2)
        
        f_2 = open(os.path.join(test_dir, 'test_xc_2.csv'), 'w')
        f_10 = open(os.path.join(test_dir, 'test_xc_10.csv'), 'w')
        f_100 = open(os.path.join(test_dir, 'test_xc_100.csv'), 'w')
        f_500 = open(os.path.join(test_dir, 'test_xc_500.csv'), 'w')
        f_1000 = open(os.path.join(test_dir, 'test_xc_1000.csv'), 'w')
        f_5000 = open(os.path.join(test_dir, 'test_xc_5000.csv'), 'w')
        f_10000 = open(os.path.join(test_dir, 'test_xc_10000.csv'), 'w')
        
        for idx in target_idx:
            row = df.iloc[idx]
            rowid, source, arch, comp, opti = row['rowid'], row['source'], row['arch'], row['comp'], row['opti']
            
            tmp_df = df[(df['arch'] == arch) & (df['comp'] != comp) & (df['opti'] != opti)]
            same_source_df = tmp_df[(tmp_df['source'] == source) & (tmp_df['rowid'] != rowid)]
            diff_source_df = tmp_df[tmp_df['source'] != source]
            
            if len(diff_source_df) < 9999: continue
            
            ## get 1 positive sample
            rowids = list(same_source_df['rowid'])
            if len(rowids) == 0: continue
            pos_rowid = random.sample(rowids, 1)[0]
            
            ## get 1 negative sample (poolsize=2)
            rowids = list(diff_source_df['rowid'])
            neg_rowids = random.sample(rowids, 1)
            write_list_to_file(f_2, [rowid, pos_rowid] + neg_rowids)
            
            ## get 9 negative samples (poolsize=10)
            neg_rowids = random.sample(rowids, 9)
            write_list_to_file(f_10, [rowid, pos_rowid] + neg_rowids)
            
            ## get 99 negative samples (poolsize=100)
            neg_rowids = random.sample(rowids, 99)
            write_list_to_file(f_100, [rowid, pos_rowid] + neg_rowids)
            
            ## get 499 negative samples (poolsize=500)
            neg_rowids = random.sample(rowids, 499)
            write_list_to_file(f_500, [rowid, pos_rowid] + neg_rowids)
            
            ## get 999 negative samples (poolsize=1000)
            neg_rowids = random.sample(rowids, 999)
            write_list_to_file(f_1000, [rowid, pos_rowid] + neg_rowids)
            
            ## get 4999 negative samples (poolsize=5000)
            neg_rowids = random.sample(rowids, 4999)
            write_list_to_file(f_5000, [rowid, pos_rowid] + neg_rowids)
            
            ## get 9999 negative samples (poolsize=10000)
            neg_rowids = random.sample(rowids, 9999)
            write_list_to_file(f_10000, [rowid, pos_rowid] + neg_rowids)

            count += 1; pbar.update()
            if count == 10000: break
            
        if count != 10000: raise Exception('XM testset: not enough samples')
        
        f_10000.close(); f_5000.close(); f_1000.close(); f_500.close(); f_100.close(); f_10.close(); f_2.close()
        pbar.close()

    # * XA (diff architecture)
    if not all(os.path.exists(os.path.join(test_dir, f'test_xa_{i}.csv')) for i in [2, 10, 100, 500, 1000, 5000, 10000]):
        count, pbar = 0, tqdm.tqdm(desc='XA testset: ', total=10000)    
        target_idx = random.sample(list(range(len(df))), len(df)//2)
        
        f_2 = open(os.path.join(test_dir, 'test_xa_2.csv'), 'w')
        f_10 = open(os.path.join(test_dir, 'test_xa_10.csv'), 'w')
        f_100 = open(os.path.join(test_dir, 'test_xa_100.csv'), 'w')
        f_500 = open(os.path.join(test_dir, 'test_xa_500.csv'), 'w')
        f_1000 = open(os.path.join(test_dir, 'test_xa_1000.csv'), 'w')
        f_5000 = open(os.path.join(test_dir, 'test_xa_5000.csv'), 'w')
        f_10000 = open(os.path.join(test_dir, 'test_xa_10000.csv'), 'w')
        
        for idx in target_idx:
            row = df.iloc[idx]
            rowid, source, arch, comp, opti = row['rowid'], row['source'], row['arch'], row['comp'], row['opti']
            
            tmp_df = df[(df['arch'] != arch) & (df['comp'] == comp) & (df['opti'] == opti)]
            same_source_df = tmp_df[(tmp_df['source'] == source) & (tmp_df['rowid'] != rowid)]
            diff_source_df = tmp_df[tmp_df['source'] != source]
            
            if len(diff_source_df) < 9999: continue
            
            ## get 1 positive sample
            rowids = list(same_source_df['rowid'])
            if len(rowids) == 0: continue
            pos_rowid = random.sample(rowids, 1)[0]
            
            ## get 1 negative sample (poolsize=2)
            rowids = list(diff_source_df['rowid'])
            neg_rowids = random.sample(rowids, 1)
            write_list_to_file(f_2, [rowid, pos_rowid] + neg_rowids)
            
            ## get 9 negative samples (poolsize=10)
            neg_rowids = random.sample(rowids, 9)
            write_list_to_file(f_10, [rowid, pos_rowid] + neg_rowids)

            ## get 99 negativev samples
            neg_rowids = random.sample(rowids, 99)
            write_list_to_file(f_100, [rowid, pos_rowid] + neg_rowids)
            
            ## get 499 negative samples
            neg_rowids = random.sample(rowids, 499)
            write_list_to_file(f_500, [rowid, pos_rowid] + neg_rowids)
            
            ## get 999 negative samples
            neg_rowids = random.sample(rowids, 999)
            write_list_to_file(f_1000, [rowid, pos_rowid] + neg_rowids)
            
            ## get 4999 negative samples
            neg_rowids = random.sample(rowids, 4999)
            write_list_to_file(f_5000, [rowid, pos_rowid] + neg_rowids)
            
            ## get 9999 negative samples
            neg_rowids = random.sample(rowids, 9999)
            write_list_to_file(f_10000, [rowid, pos_rowid] + neg_rowids)

            count += 1; pbar.update()
            if count == 10000: break
            
        if count != 10000: raise Exception('XM testset: not enough samples')
        
        f_10000.close(); f_5000.close(); f_1000.close(); f_500.close(); f_100.close(); f_10.close(); f_2.close()
        pbar.close()

    # * XM (diff architecture + diff compiler + diff optimization)
    # * different pool size: 2, 10, 100, 500, 1000, 5000, 10000
    if not all(os.path.exists(os.path.join(test_dir, f'test_xm_{i}.csv')) for i in [2, 10, 100, 500, 1000, 5000, 10000]):
        count, pbar = 0, tqdm.tqdm(desc='XM testset (7 kinds of poolsize): ', total=10000)    
        target_idx = random.sample(list(range(len(df))), len(df)//2)
        
        f_2 = open(os.path.join(test_dir, 'test_xm_2.csv'), 'w')
        f_10 = open(os.path.join(test_dir, 'test_xm_10.csv'), 'w')
        f_100 = open(os.path.join(test_dir, 'test_xm_100.csv'), 'w')
        f_500 = open(os.path.join(test_dir, 'test_xm_500.csv'), 'w')
        f_1000 = open(os.path.join(test_dir, 'test_xm_1000.csv'), 'w')
        f_5000 = open(os.path.join(test_dir, 'test_xm_5000.csv'), 'w')
        f_10000 = open(os.path.join(test_dir, 'test_xm_10000.csv'), 'w')
        
        for idx in target_idx:
            row = df.iloc[idx]
            rowid, source, arch, comp, opti = row['rowid'], row['source'], row['arch'], row['comp'], row['opti']
            
            tmp_df = df[(df['arch'] != arch) & (df['comp'] != comp) & (df['opti'] != opti)]
            same_source_df = tmp_df[(tmp_df['source'] == source) & (tmp_df['rowid'] != rowid)]
            diff_source_df = tmp_df[tmp_df['source'] != source]
            
            if len(diff_source_df) < 9999: continue
            
            ## get 1 positive sample
            rowids = list(same_source_df['rowid'])
            if len(rowids) == 0: continue
            pos_rowid = random.sample(rowids, 1)[0]
            
            ## get 1 negative sample (poolsize=2)
            rowids = list(diff_source_df['rowid'])
            neg_rowids = random.sample(rowids, 1)
            write_list_to_file(f_2, [rowid, pos_rowid] + neg_rowids)

            ## get 9 negative sample (poolsize=10)
            neg_rowids = random.sample(rowids, 9)
            write_list_to_file(f_10, [rowid, pos_rowid] + neg_rowids)
            
            ## get 99 negative samples (poolsize=100)
            neg_rowids = random.sample(rowids, 99)
            write_list_to_file(f_100, [rowid, pos_rowid] + neg_rowids)

            ## get 499 negative samples (poolsize=500)
            neg_rowids = random.sample(rowids, 499)
            write_list_to_file(f_500, [rowid, pos_rowid] + neg_rowids)

            ## get 999 negative samples (poolsize=1000)
            neg_rowids = random.sample(rowids, 999)
            write_list_to_file(f_1000, [rowid, pos_rowid] + neg_rowids)

            ## get 4999 negative samples (poolsize=5000)
            neg_rowids = random.sample(rowids, 4999)
            write_list_to_file(f_5000, [rowid, pos_rowid] + neg_rowids)

            ## get 9999 negative samples (poolsize=10000)
            neg_rowids = random.sample(rowids, 9999)
            write_list_to_file(f_10000, [rowid, pos_rowid] + neg_rowids)

            count += 1; pbar.update()
            if count == 10000: break
        
        if count != 10000: raise Exception('XM testset: not enough samples')
        
        f_10000.close(); f_5000.close(); f_1000.close(); f_500.close(); f_100.close(); f_10.close(); f_2.close()
        pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', type=str)
    args = parser.parse_args()
    
    build_search_tasks(db_dir=args.db_dir)
    
    
if __name__ == '__main__':
    main()
    
