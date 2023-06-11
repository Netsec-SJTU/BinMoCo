# @Time: 2022.12.26 20:39
# @Author: Bolun Wu

import re


class InsNormalizer(object):
    """
        ins_record is defined in `code/analyzer/radare2.py`
    """
    def __init__(self):
        return
            
    def parse(self, ins_record: list):
        _, ins, opcode_type, operands_type = ins_record
        ins_split = ins.split(' ')
        opcode, operands = ins_split[0], ' '.join(ins_split[1:])
        operands = self.clean(operands)

        if len(operands) > len(operands_type):
            operands_type += ['ope_unk' for _ in range(len(operands) - len(operands_type))]
        if len(operands) < len(operands_type):
            for _ in range(len(operands_type) - len(operands)): operands_type.pop()
                    
        tokens, tokens_type, pos = self.split_normalize(operands, operands_type)
        tokens = [opcode] + tokens
        tokens_type = [opcode_type] + tokens_type
        pos = [0] + pos
        assert len(tokens) == len(tokens_type) == len(pos)
        return tokens, tokens_type, pos
    
    def clean(self, operands: str):
        operands = operands.strip()
        operands = operands.replace('ptr ','')
        operands = operands.replace('offset ','')
        operands = operands.replace('xmmword ','')
        operands = operands.replace('dword ','')
        operands = operands.replace('qword ','')
        operands = operands.replace('word ','')
        operands = operands.replace('byte ','')
        operands = operands.replace('short ','')
        operands = operands.replace('!', '')
        
        operands = operands.split(', ')
        operands = list(filter(lambda x: x!='', operands))
        
        # deal with: stp x21, x22, [sp, 0x20]
        left_pos, right_pos = None, None
        for i, ope in enumerate(operands):
            if '[' in ope and ']' not in ope:
                left_pos = i
            if ']' in ope and '[' not in ope:
                right_pos = i
        if left_pos is not None and right_pos is not None and left_pos < right_pos:
            operands = operands[:left_pos] + [', '.join(operands[left_pos: right_pos+1])] + operands[right_pos+1:]
            
        return operands
    
    def split_normalize(self, operands: list, operands_type: list):    
        p_operands, p_operands_type, p_pos = [], [], []
        
        for i, (operand, operand_type) in enumerate(zip(operands, operands_type)):
            split = re.split('(\W)', operand)
            split = list(filter(lambda x: x.strip()!='', split))
            for j, s in enumerate(split):
                change_type = False
                if re.match(r'0x[0-9a-f]+', s):
                    num = int(s, 16)
                    
                    if num >= int('0x10000', 16): # at least 5 digits: address
                        split[j] = '<addr>'
                        change_type = True
                    elif num <= int('0xfff', 16): # less than 4 digits: immediates
                        split[j] = str(num) # ! should save as str, otherwise 13 and '13' is different
                    else: # big const
                        split[j] = '<const>'

                p_pos.append(i+1)
                p_operands.append(split[j])
                if operand_type == 'imm' and change_type: p_operands_type.append('mem')
                else: p_operands_type.append(operand_type)
                
        return p_operands, p_operands_type, p_pos
    
    ## ! below are parsers from SOTA works ##
    
    def parse_proposed_in_jTrans(self, ins_record: list):
        """
        Ref: jTrans: Jump-Aware Transformer for Binary Code Similarity Detection
        """
        ins_addr, ins, opcode_type, _ = ins_record
        ins_addr = int(ins_addr)
        
        ins_split = ins.split(' ')
        opcode, operands = ins_split[0], ' '.join(ins_split[1:])
        operands = self.clean(operands)

        addr_operand_idx, jmp_addr = None, None
        
        # normalize operands
        norm_operands = []
        for i, operand in enumerate(operands):
            split = re.split('(\W)', operand)
            split = list(filter(lambda x: x.strip()!='', split))
            for j, s in enumerate(split):
                if re.match(r'0x[0-9a-f]+', s):
                    num = int(s, 16)
                    if num >= int('0x10000', 16): # at least 5 digits: address
                        split[j] = '<addr>'
                        if 'jmp' in opcode_type:
                            addr_operand_idx, jmp_addr = i, num
                    elif num <= int('0xf', 16): # less than 2 digits: immediates
                        split[j] = str(num) # ! should save as str, otherwise 13 and '13' is different
                    else: # big const
                        split[j] = '<const>'

            norm_operands.append(''.join(split))
        
        ins = [opcode] + norm_operands
        if 'jmp' in opcode_type and addr_operand_idx is not None: 
            addr_operand_idx += 1
        return ins_addr, ins, opcode_type, (addr_operand_idx, jmp_addr)
        
    def parse_proposed_in_GMN(self, ins_record: list):
        """
        Ref: Graph Matching Networks for Learning the Similarity of Graph Structured Objects
        only consider opcode
        """
        _, ins, _, _ = ins_record
        opcode = ins.split(' ')[0]
        return opcode
        
    def parse_proposed_in_SAFE(self, ins_record: list):
        """
        Ref: SAFE: Self-Attentive Function Embeddings for Binary Similarity
        the same as BAR 2019 paper
        """
        return self.parse_proposed_in_BAR2019(ins_record)
        
    def parse_proposed_in_BAR2019(self, ins_record: list):
        """
        Ref: Investigating Graph Embedding Neural Networks with Unsupervised Features Extraction for Binary Analysis
        regard each instruction as a token
        """
        _, ins, _, operands_type = ins_record
        ins_split = ins.split(' ')
        opcode, operands = ins_split[0], ' '.join(ins_split[1:])
        operands = self.clean(operands)
        
        if len(operands) > len(operands_type):
            operands_type += ['ope_unk' for _ in range(len(operands) - len(operands_type))]
        if len(operands) < len(operands_type):
            for _ in range(len(operands_type) - len(operands)): operands_type.pop()
        
        tokens, _, pos = self.split_normalize(operands, operands_type)
        
        # merge tokens belonging to one operand
        operands, last_pos = [], 1
        token_for_one_operand = []
        for token, _pos in zip(tokens, pos):
            if _pos == last_pos: 
                token_for_one_operand.append(token)
            else:
                token_for_one_operand = ''.join(token_for_one_operand)
                operands.append(token_for_one_operand)
                token_for_one_operand = [token]
                last_pos = _pos
        if len(token_for_one_operand) > 0:
            token_for_one_operand = ''.join(token_for_one_operand)
            operands.append(token_for_one_operand)
            
        ins = [opcode] + operands
        ins = '_'.join(ins)
        return ins
    
    