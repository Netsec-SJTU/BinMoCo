# @Time: 2022.12.8 17:32
# @Author: Bolun Wu

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import multiprocessing
import random
import sqlite3
from multiprocessing.dummy import Pool as ThreadPool

import connectorx as cx
import tqdm
from analyzer.radare2 import Radare2FunctionAnalyzer
from constants import log_dir, seed
from loguru import logger


class DataFactory(object):
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_cg_path = db_path.replace('.sqlite', '_cg.sqlite')
        assert self.db_path != self.db_cg_path
    
    @staticmethod
    def analyze_file(args):
        global pool_sem
        
        (filepath, db_path, 
        project, binary_name, 
        arch, bits, compiler, compiler_version, opti) = args
        
        analyzer = Radare2FunctionAnalyzer(filepath=filepath, debug_info=True)     
        p = ThreadPool(1)
        async_res = p.apply_async(analyzer.analyze_all)
            
        try:
            _ = async_res.get(180) # run 'aaa' for 2 minutes, invalid
            
        except:
            logger.error(f'{filepath}: radare2 command "aaa" timeout.')
            analyzer.close()
            return -1

        try:
            func_cfg_var = analyzer.get_all_function_cfg_var(ignore_no_source=True)
            for func_name, cfg_var in func_cfg_var.items():
                insert_contents = (os.path.basename(filepath), project, binary_name,
                                arch, bits, compiler, compiler_version, opti, 
                                func_name, json.dumps(cfg_var), len(cfg_var['blocks']), cfg_var['src'])
                DataFactory.insert_db(db_path, pool_sem, insert_contents)
            
            if len(func_cfg_var) == 0:
                logger.warning(f'{filepath}: 0 valid functions.')
            else:
                logger.info(f'{filepath}: success with {len(func_cfg_var)} valid functions.')

        except Exception as e:
            logger.error(f'{filepath}: {repr(e)}.')
            analyzer.close()
            return -1
        
        p.close()
        analyzer.close()
        return 0

    def create_db(self):
        print('Database creation...')
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS functions (
            file_name text, project text, binary_name text,
            architecture text, bits text, compiler text, compiler_version text, optimization text,
            function_name text, graph text, num_block int, source_line text)''')
        conn.commit()
        conn.close()
        
    @staticmethod
    def insert_db(db_path, pool_sem, insert_contents):
        (file_name, project, binary_name, 
         arch, bits, compiler, compiler_version, opti, 
         func_name, graph, num_block, source_line) = insert_contents
        
        pool_sem.acquire()
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            '''INSERT INTO functions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (file_name, project, binary_name,
             arch, bits, compiler, compiler_version, opti,
             func_name, graph, num_block, source_line))
        conn.commit()
        conn.close()
        
        pool_sem.release()

    @staticmethod
    def analyze_cg_file(args):
        global pool_sem
        
        (filepath, db_path) = args
        analyzer = Radare2FunctionAnalyzer(filepath=filepath, debug_info=False)
        p = ThreadPool(1)
        async_res = p.apply_async(analyzer.analyze_all)
    
        try:
            _ = async_res.get(180) # run 'aaa' for 3 minutes, invalid
            
        except:
            logger.error(f'{filepath}: radare2 command "aaa" timeout.')
            analyzer.close()
            return -1
    
        try:
            raw_cg = analyzer.analyze_CG()
            insert_contents = (os.path.basename(filepath), json.dumps(raw_cg))
            DataFactory.insert_cg_db(db_path, pool_sem, insert_contents)
            logger.info(f'{filepath}: success CG extraction.')
        except Exception as e:
            logger.error(f'{filepath}: {repr(e)}.')
            analyzer.close()
            return -1
        
        p.close()
        analyzer.close()
        return 0
        
    def create_cg_db(self):
        print('Database CG creation...')
        conn = sqlite3.connect(self.db_cg_path)
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS binary_cg (
                file_name text, cg text)''')
        conn.commit()
        conn.close()
        
    @staticmethod
    def insert_cg_db(db_path, pool_sem, insert_contents):
        file_name, cg = insert_contents
        
        pool_sem.acquire()
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''INSERT INTO binary_cg VALUES (?,?)''', (file_name, cg))
        conn.commit()
        conn.close()
        
        pool_sem.release()

    @staticmethod
    def analyze_func_and_cg(args):
        global pool_sem
        global pool_sem_cg
        
        (filepath, db_path, db_cg_path,
         project, binary_name,
         arch, bits, compiler, compiler_version, opti) = args
        
        analyzer = Radare2FunctionAnalyzer(filepath=filepath, debug_info=False)
        p = ThreadPool(1)
        async_res = p.apply_async(analyzer.analyze_all)
        
        try:
            _ = async_res.get(1800) # run 'aaa' for 30 minutes, invalid
        except:
            logger.error(f'{filepath}: radare2 command "aaa" timeout.')
            analyzer.close()
            return -1
        
        # logger.info(f'{filepath}: aa done.')
        
        try:
            # logger.info(f'{filepath}: start function and cg extraction. about {len(analyzer.list_functions())} functions in total.')
            
            func_name_to_cfg = analyzer.get_all_function_cfg(ignore_no_source=False, with_tqdm=False)
            for func_name, cfg in func_name_to_cfg.items():
                insert_contents = (
                    os.path.basename(filepath), project, binary_name,
                    arch, bits, compiler, compiler_version, opti,
                    func_name, json.dumps(cfg), len(cfg['blocks']), ''
                )
                DataFactory.insert_db(db_path, pool_sem, insert_contents)

            raw_cg = analyzer.analyze_CG()
            insert_contents = (os.path.basename(filepath), json.dumps(raw_cg))
            DataFactory.insert_cg_db(db_cg_path, pool_sem_cg, insert_contents)
            logger.info(f'{filepath}: success with {len(func_name_to_cfg)} valid functions and CG extraction.')        
        
        except Exception as e:
            logger.error(f'{filepath}: {repr(e)}.')
            analyzer.close()
            return -1
        
        p.close()
        analyzer.close()
        return 0
        
class BinKitFactory(DataFactory):
    
    def scan_directory(self, dirpaths):
        """we only consider two versions of compiler:
        - gcc-6.4.0
        - clang-6.0
        """
        if isinstance(dirpaths, str): dirpaths = [dirpaths]
        filepaths = []
        
        for dirpath in dirpaths:
            for root, _, paths in os.walk(dirpath):
                paths = list(filter(lambda x: 'gcc-6.4.0' in x or 'clang-6.0' in x, paths))
                filepaths.extend(list(map(lambda x: os.path.join(root, x), paths)))
        
        return filepaths
    
    def convert_filename_to_args(self, filename):
        # e.g., /.../which-2.21_gcc-4.9.4_mipseb_32_O0_which.elf
        filename = os.path.splitext(os.path.basename(filename))[0]
        filename = filename.split('_')
        project, compiler, arch, bits, opti = filename[:5]
        binary_name = '_'.join(filename[5:])
        compiler, compiler_version = compiler.split('-')
        return project, binary_name, arch, bits, compiler, compiler_version, opti
    
    def remove_repeats(self, filepaths):
        query = '''SELECT DISTINCT file_name FROM functions'''
        parsed_filenames = list(cx.read_sql(f'sqlite://{self.db_path}', query)['file_name'])
        cleaned = list(filter(lambda x: os.path.basename(x) not in parsed_filenames, filepaths))
        return cleaned
    
    def remove_cg_repeats(self, filepaths):
        query = '''SELECT DISTINCT file_name FROM binary_cg'''
        parsed_filenames = list(cx.read_sql(f'sqlite://{self.db_cg_path}', query)['file_name'])
        cleaned = list(filter(lambda x: os.path.basename(x) not in parsed_filenames, filepaths))
        return cleaned
    
    def build_db(self, source_dirpaths, num_proc=os.cpu_count()):

        # logger
        logger.remove()
        logger.add(os.path.join(log_dir, '{time}.log'))

        # * write db semaphore
        global pool_sem
        pool_sem = multiprocessing.BoundedSemaphore(value=1)
        
        self.create_db()
        
        filepaths = self.scan_directory(source_dirpaths)
        print(f'find {len(filepaths)} files in {source_dirpaths}.')
        filepaths = self.remove_repeats(filepaths)
        print(f'find {len(filepaths)} not parsed yet.')
        
        random.seed(seed)
        random.shuffle(filepaths)
        
        p_args = []
        for path in filepaths:
            _args = self.convert_filename_to_args(path)
            _args = [path, self.db_path] + list(_args)
            p_args.append(tuple(_args))
        
        print(f'using {num_proc} cpu cores.')
        pool = multiprocessing.Pool(processes=num_proc)
        _ = list(tqdm.tqdm(pool.imap_unordered(DataFactory.analyze_file, p_args), total=len(p_args), ncols=100))
        pool.close()
        pool.join()
        
    def build_cg_db(self, source_dirpaths, num_proc=os.cpu_count()):
        # logger
        logger.remove()
        logger.add(os.path.join(log_dir, '{time}.log'))

        # * write db semaphore
        global pool_sem
        pool_sem = multiprocessing.BoundedSemaphore(value=1)
        
        self.create_cg_db()
        
        filepaths = self.scan_directory(source_dirpaths)
        print(f'find {len(filepaths)} files in {source_dirpaths}.')
        filepaths = self.remove_cg_repeats(filepaths)
        print(f'find {len(filepaths)} not parsed yet.')
        
        random.seed(seed)
        random.shuffle(filepaths)
        
        p_args = [[path, self.db_cg_path] for path in filepaths]
        
        print(f'using {num_proc} cpu cores.')
        pool = multiprocessing.Pool(processes=num_proc)
        _ = list(tqdm.tqdm(pool.imap_unordered(DataFactory.analyze_cg_file, p_args), total=len(p_args), ncols=100))
        pool.close()
        pool.join()


class Sec22Factory(DataFactory):
    
    def scan_directory(self, dirpaths):
        """we only consider two versions of compiler
        - gcc-9
        - clang-9
        """
        if isinstance(dirpaths, str): dirpaths = [dirpaths]
        filepaths = []
        
        for dirpath in dirpaths:
            if os.path.basename(dirpath) == 'Dataset-1':
                for proj_name in ['unrar', 'zlib', 'curl', 'nmap', 'openssl', 'clamav', 'z3']:
                    proj_dirpath = os.path.join(dirpath, proj_name)
                    for filename in os.listdir(proj_dirpath):
                        if 'gcc-9' not in filename and 'clang-9' not in filename: continue
                        filepath = os.path.join(proj_dirpath, filename)
                        assert os.path.exists(filepath)
                        filepaths.append(filepath)
            else:
                for root, _, paths in os.walk(dirpath):
                    paths = list(filter(lambda x: 'gcc-9' in x or 'clang-9' in x, paths))
                    filepaths.extend(list(map(lambda x: os.path.join(root, x), paths)))
        
        return filepaths

    def convert_filename_to_args(self, filename):
        # e.g., /.../curl/mips32-gcc-9-O1_curl
        parts_of_path = filename.split('/')
        project, filename = parts_of_path[-2:]
        env_configs, binary_name = filename.split('_')
        
        env_configs = env_configs.split('-')
        compiler, compiler_version, opti = env_configs[1:4]
        
        architecture = env_configs[0]
        if 'arm' in architecture:
            arch = 'arm'
            if '32' in architecture: bits = '32'
            elif '64' in architecture: bits = '64'
        elif 'mips' in architecture:
            arch = 'mips'
            if '32' in architecture: bits = '32'
            elif '64' in architecture: bits = '64'
        elif 'x' in architecture: # x86, x64
            arch = 'x86'
            if '86' in architecture: bits = '32'
            elif '64' in architecture: bits = '64'
        
        return project, binary_name, arch, bits, compiler, compiler_version, opti
        
    def remove_repeats(self, filepaths):
        query = '''select distinct file_name from functions'''
        parsed_func_fnames = list(cx.read_sql(f'sqlite://{self.db_path}', query)['file_name'])
        
        query = '''select distinct file_name from binary_cg'''
        parsed_cg_fnames = list(cx.read_sql(f'sqlite://{self.db_cg_path}', query)['file_name'])
        
        parsed_fnames = list(set(parsed_func_fnames).intersection(set(parsed_cg_fnames)))
        cleaned = list(filter(lambda x: os.path.basename(x) not in parsed_fnames, filepaths))
        
        return cleaned
    
    def build_db(self, source_dirpaths, num_proc=os.cpu_count()):
        # logger
        logger.remove()
        logger.add(os.path.join(log_dir, '{time}.log'))
        
        # * write db semaphore
        global pool_sem, pool_sem_cg
        pool_sem = multiprocessing.BoundedSemaphore(value=1)
        pool_sem_cg = multiprocessing.BoundedSemaphore(value=1)
        
        self.create_db()
        self.create_cg_db()
        
        filepaths = self.scan_directory(source_dirpaths)
        print(f'find {len(filepaths)} files in {source_dirpaths}.')
        filepaths = self.remove_repeats(filepaths)
        print(f'find {len(filepaths)} not parsed yet.')
        
        # random.seed(seed)
        # random.shuffle(filepaths)
        
        p_args = []
        for path in filepaths:
            _args = self.convert_filename_to_args(path)
            _args = [path, self.db_path, self.db_cg_path] + list(_args)
            p_args.append(tuple(_args))
        
        print(f'using {num_proc} cpu cores.')
        pool = multiprocessing.Pool(processes=num_proc)
        _ = list(tqdm.tqdm(pool.imap_unordered(DataFactory.analyze_func_and_cg, p_args), total=len(p_args), ncols=100))
        pool.close()
        pool.join()
    

class BinKit_v2Factory(DataFactory):
    
    def scan_directory(self, dirpaths):
        """we only consider 2 compiler version:
        - gcc-9.4.0
        - clang-9.0
        we only consider 5 optimization levels:
        - O0, O1, O2, O3, Os
        we only consider 6 architectures:
        - x86, x86_64, arm, arm64, mips, mips64
        """
        if isinstance(dirpaths, str): dirpaths = [dirpaths]
        filepaths = []
        for dirpath in dirpaths:
            for root, _, paths in os.walk(dirpath):
                # filter compiler version
                paths = list(filter(lambda x: 'gcc-9.4.0' in x or 'clang-9.0' in x, paths))
                # filter optimization
                paths = list(filter(lambda x: 'Ofast' not in x, paths))
                # filter architecture
                paths = list(filter(lambda x: 'mipseb' not in x, paths))
                filepaths.extend(list(map(lambda x: os.path.join(root, x), paths)))
        return filepaths

    def convert_filename_to_args(self, filename):
        # e.g., /.../which-2.21_gcc-4.9.4_mipseb_32_O0_which
        filename = os.path.basename(filename)
        filename = filename.split('_')
        project, compiler, arch, bits, opti = filename[:5]
        binary_name = '_'.join(filename[5:])
        compiler, compiler_version = compiler.split('-')
        return project, binary_name, arch, bits, compiler, compiler_version, opti
    
    def remove_repeats(self, filepaths):
        query = '''select distinct file_name from functions'''
        parsed_func_fnames = list(cx.read_sql(f'sqlite://{self.db_path}', query)['file_name'])
        
        query = '''select distinct file_name from binary_cg'''
        parsed_cg_fnames = list(cx.read_sql(f'sqlite://{self.db_cg_path}', query)['file_name'])
        
        parsed_fnames = list(set(parsed_func_fnames).intersection(set(parsed_cg_fnames)))
        cleaned = list(filter(lambda x: os.path.basename(x) not in parsed_fnames, filepaths))
        
        return cleaned
    
    def build_db(self, source_dirpaths, num_proc=os.cpu_count()):
        # logger
        logger.remove()
        logger.add(os.path.join(log_dir, '{time}.log'))
        
        # * write db semaphore
        global pool_sem, pool_sem_cg
        pool_sem = multiprocessing.BoundedSemaphore(value=1)
        pool_sem_cg = multiprocessing.BoundedSemaphore(value=1)
        
        self.create_db()
        self.create_cg_db()

        filepaths = self.scan_directory(source_dirpaths)
        print(f'find {len(filepaths)} files in {source_dirpaths}.')
        filepaths = self.remove_repeats(filepaths)
        print(f'find {len(filepaths)} not parsed yet.')

        p_args = []
        for path in filepaths:
            _args = self.convert_filename_to_args(path)
            _args = [path, self.db_path, self.db_cg_path] + list(_args)
            p_args.append(tuple(_args))

        print(f'using {num_proc} cpu cores.')
        pool = multiprocessing.Pool(processes=num_proc)
        _ = list(tqdm.tqdm(pool.imap_unordered(DataFactory.analyze_func_and_cg, p_args), total=len(p_args), ncols=100))
        pool.close()
        pool.join()
        
    
def test_build_db_binkit():
    factory = BinKitFactory(db_path='demo.sqlite')
    factory.build_cg_db(['/home/wubolun/data/bcsd/binkit/normal_dataset/',
                      '/home/wubolun/data/bcsd/binkit/gnu_debug_sizeopt'], num_proc=1)


def test_build_db_sec22():
    factory = Sec22Factory(db_path='demo.sqlite')
    factory.build_db(['/home/wubolun/data/bcsd/security22_review/Dataset-1'], num_proc=8)


def test_build_db_binkit_v2():
    factory = BinKit_v2Factory(db_path='demo.sqlite')
    factory.build_db(['/home/wubolun/data/bcsd/binkit_2.0/BinKit_normal'], 
                     num_proc=4)


if __name__ == '__main__':
    # test_build_db_binkit()
    # test_build_db_sec22()
    test_build_db_binkit_v2()
    
    
    