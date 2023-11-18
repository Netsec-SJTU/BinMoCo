import json
import os
import re


class dotdict(dict):
    """dot.notation access to dictionary attributes
    ref: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    
    # ! pickle.dump ERROR: TypeError: 'NoneType' object is not callable
    # ref: https://stackoverflow.com/questions/2049849/why-cant-i-pickle-this-object
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


def counter_to_dict(counter):
    sorted_keys = sorted(counter, key=counter.get, reverse=True)
    return {k: counter[k] for k in sorted_keys}


def fix_json_string(s):
    """fix invalid JSON string which fails on `json.loads(s)`

    Args:
        s (_type_): _description_

    Returns:
        _type_: _description_
    """
    while True:
        try:
            _ = json.loads(s)
            break
        except Exception as e:
            e = str(e)
            pos_id = re.search(r'\(char (\d+)\)', e).span()
            pos_id = e[pos_id[0]: pos_id[1]]
            pos_id = int(pos_id.split()[-1].rstrip(')'))
            
            s = list(s)
            s[pos_id] = ','
            s = ''.join(s)
            
    return s


def is_smaller_hex(a, b):
    return int(a, 16) < int(b, 16)


def is_same_hex(a, b):
    return int(a, 16) == int(b, 16)


def is_bigger_hex(a, b):
    return int(a, 16) > int(b, 16)


def count_code_line():
    res = 0
    code_dirpath = os.path.dirname(__file__)
    for root, _, paths in os.walk(code_dirpath):
        for path in paths:
            if path.endswith('.py'):
                filepath = os.path.join(root, path)
                line = len(open(filepath, 'r').readlines())
                res += line
    return res


def count_project_functions(group_csv_path):
    import pandas as pd
    df = pd.read_csv(group_csv_path)
    func_count = {}
    for i, row in df.iterrows():
        proj = row['project']
        if proj not in func_count: func_count[proj] = 0
        rowids = row['rowids']
        rowids = rowids.split()
        func_count[proj] += len(rowids)
    return func_count

    
if __name__ == '__main__':
    
    t = '[{"opcode":"jmp rax","disasm":"jmp rax","pseudo":"goto loc_rax","description":"jump","srcs":["name":"rax","type":"reg"],"mnemonic":"jmp","mask":"ffff","esil":"rax,rip,=","sign":false,"prefix":0,"id":172,"opex":{"operands":[{"size":8,"rw":1,"type":"reg","value":"rax"}],"modrm":true},"addr":4198229,"bytes":"ffe0","size":2,"type":"rjmp","esilcost":0,"reg":"rax","scale":0,"refptr":0,"cycles":2,"failcycles":0,"delay":0,"stackptr":0,"direction":"exec","family":"cpu"}]'
    res = fix_json_string(t)
    print(res)
    
    print(count_code_line())
    