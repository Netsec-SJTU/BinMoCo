# @Time: 2022.12.2 14:56
# @Author: Bolun Wu

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from collections import defaultdict

import networkx as nx
import r2pipe
import tqdm
from utils import fix_json_string


class Radare2Analyzer(object):
    
    def __init__(self, filepath):
        """Base Radare2 Analyzer

        Args:
            filepath (str): absolute path of file to be analyzed
        """
        
        self.filepath = filepath
        self.r2 = r2pipe.open(filepath, flags=['-2']) # -2 signifies disable stderr
        
        self.__initialize()
    
    def __initialize(self):
        self.meta = self.r2.cmdj('ij')

    def analyze_all(self):
        self.r2.cmd('aa')
    
    def analyze_all_deeper(self):
        self.r2.cmd('aaa')

    def analyze_CG(self):
        """analyze global call graph with `agC`"""
        raw_cg = self.r2.cmdj('agCj')
        return raw_cg
        
    @property
    def arch(self):    
        return self.meta['bin']['arch']
    
    @property
    def bits(self):
        return self.meta['bin']['bits']
    
    @property
    def endian(self):
        return self.meta['bin']['endian']

    def get_clean_inst(self, inst_addr):
        self.r2.cmd(f's {inst_addr}')
        inst_json_str = self.r2.cmd('aoj1')
        inst_json_str = fix_json_string(inst_json_str)
        inst_json = json.loads(inst_json_str)
        
        if len(inst_json) == 0: return # instruction disassemble failed
        inst_json = inst_json[0]
        if inst_json['opcode'] == 'invalid': return # invalid instruction
        if 'opex' not in inst_json: return # invalid instruction
        
        operands_type = [x['type'] for x in inst_json['opex']['operands']]
        
        clean_inst_json = [
            inst_json['addr'],    # addr
            inst_json['opcode'],  # ins
            inst_json['type'],    # mnemonic type
            operands_type         # operands type
        ]
        
        return clean_inst_json

    def get_source_line(self):
        result = self.r2.cmd('id | grep ^0x.*/.*').split('\n')[:-1] # the last is empty
        if len(result) == 0: return None # no debug information found
        
        addr_to_file_line = defaultdict(str)
        for res in result:
            addr, filepath, line = res.split('\t')
            addr = int(addr, 16)
            addr_to_file_line[addr] = f'{os.path.basename(filepath)}@{line}'
        
        return addr_to_file_line   

    def close(self):
        self.r2.quit()


class Radare2FunctionAnalyzer(Radare2Analyzer):

    def __init__(self, filepath, debug_info=True):
        super(Radare2FunctionAnalyzer, self).__init__(filepath)
        self.debug_info = debug_info
        if debug_info: 
            self.addr_to_file_line = self.get_source_line()
            assert self.addr_to_file_line != None

    def has_source_line(self, function):
        return self.addr_to_file_line[function['offset']] != ''
    
    def list_functions(self):
        """list functions using Radare2 `afl`

        Returns:
            list: list of function(JSON)
        """
        
        # ! fix bug: segmentation fault occurs when running `aflj` if too many functions
        # try: functions = self.r2.cmdj('aflj')
        # except: functions = []
        # return functions

        res = self.r2.cmd('afl')
        res = res.split('\n')[:-1] # the last is empty
        functions = []
        for r in res:
            r = r.split()
            functions.append({'offset': r[0], 'name': r[-1]})
        return functions

    def get_function_imported_functions(self, function):
        """get imported functions used by this function"""
        start_addr = function['offset']
        
        # * seek function start
        self.r2.cmd(f's {start_addr}')

        # sub call graph
        cg = self.r2.cmdj('agcj')

        imp_funcs = []
        if len(cg) == 0: return imp_funcs
        
        for func in cg[0]['imports']:
            if not func.startswith('sym.imp.'): continue
            imp_funcs.append(func[8:])
        return imp_funcs
    
    def get_all_function_cfg(self, ignore_no_source=False, with_tqdm=False):
        functions = self.list_functions()
        if ignore_no_source:
            functions = [x for x in functions if self.has_source_line(x)]
        # ignore .imp. and fcn.xxx and entry functions
        def __filter(x):
            if x['name'].startswith('fcn.') or \
               '.imp.' in x['name'] or \
               'entry' in x['name']: return False
            return True
        functions = list(filter(__filter, functions))
        
        func_name_to_cfg = dict()
        if with_tqdm:
            pbar = tqdm.tqdm(functions)
            for func in pbar:
                pbar.set_description(func['name'])
                cfg, _= self.get_function_cfg(func)
                if cfg: func_name_to_cfg[func['name']] = cfg
            pbar.close()
        else:
            for func in functions:
                cfg, _ = self.get_function_cfg(func)
                if cfg: func_name_to_cfg[func['name']] = cfg
        return func_name_to_cfg
        
    def get_function_cfg(self, function, return_networkx=False):
        """get control flow graph (CFG) given a function

        Args:
            function (dict): function in the list obtained by `self.list_functions()`

        Returns:
            dict: CFG dict {'blocks', 'cfg'}
        """
        
        start_addr = function['offset']

        # * seek function start
        self.r2.cmd(f's {start_addr}')

        # * get CFG
        if return_networkx: nx_cfg = nx.DiGraph()
        else: nx_cfg = None
        
        full_cfg = {
            'addr': start_addr,
            'name': function['name'],
            'blocks': {}, 
            'cfg': []
        }
        
        if self.debug_info:
            full_cfg['src'] = self.addr_to_file_line[start_addr]

        cfg = self.r2.cmdj('agfj')
        if len(cfg) == 0: return None, None
        
        cfg = cfg[0]
        
        for block in cfg['blocks']:
            ## block start address
            block_addr = block['offset'] # use as dict
            
            ## block instructions
            block_insts = []
            for inst in block['ops']:
                inst_addr = inst['offset']
                clean_inst = self.get_clean_inst(inst_addr=inst_addr)
                if clean_inst: block_insts.append(clean_inst)
            
            ## add block into `full_cfg`
            full_cfg['blocks'][block_addr] = block_insts
            if return_networkx: nx_cfg.add_node(block_addr)
            
            if 'jump' in block and 'fail' in block: ## conditional jump
                true_jump_addr = block['jump']
                fail_jump_addr = block['fail']
                full_cfg['cfg'].append([block_addr, true_jump_addr, 1]) # true jump (1)
                full_cfg['cfg'].append([block_addr, fail_jump_addr, 0]) # false jump (0)
                if return_networkx:
                    nx_cfg.add_edge(block_addr, true_jump_addr)
                    nx_cfg.add_edge(block_addr, fail_jump_addr)
                
            elif 'jump' in block: ## unconditional jump
                jump_addr = block['jump']
                full_cfg['cfg'].append([block_addr, jump_addr, 2]) # unconditional (2)
                if return_networkx:
                    nx_cfg.add_edge(block_addr, jump_addr)
            
            elif 'fail' in block: ## unconditional jump
                jump_addr = block['fail']
                full_cfg['cfg'].append([block_addr, jump_addr, 2])
                if return_networkx:
                    nx_cfg.add_edge(block_addr, jump_addr)

            else: ## no jump
                pass
                    
        return full_cfg, nx_cfg
       