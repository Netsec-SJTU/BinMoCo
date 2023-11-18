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
