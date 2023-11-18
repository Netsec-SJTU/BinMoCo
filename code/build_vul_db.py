import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json

from analyzer.radare2 import Radare2FunctionAnalyzer


def extract_all_functions(binary_path, save_dir):

    analyzer = Radare2FunctionAnalyzer(filepath=binary_path, debug_info=False)
    analyzer.analyze_all()
    
    # extract function cfg
    func_name_to_cfg = analyzer.get_all_function_cfg(ignore_no_source=False, with_tqdm=True)
    with open(os.path.join(save_dir, 'func_cfgs.json'), 'w') as f:
        json.dump(func_name_to_cfg, f, indent=1)
    
    # extract cg
    cg = analyzer.analyze_CG()
    with open(os.path.join(save_dir, 'cg.json'), 'w') as f:
        json.dump(cg, f, indent=1)
    
    # save meta
    filename = os.path.basename(binary_path)
    arch = filename.split('_')[-1]
    meta = {
        'filename': os.path.basename(binary_path),
        'arch': arch,
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
        

def main():
    tar_dir = '/home/user/data/bcsd/security22_review/Dataset-Vulnerability'
    save_dir = 'database/vul_libcrypto.so.1.0.0'
    for filename in os.listdir(tar_dir):
        _save_dir = os.path.join(save_dir, filename)
        os.makedirs(_save_dir, exist_ok=True)
        print(f'extracting {filename}...')
        filepath = os.path.join(tar_dir, filename)
        extract_all_functions(filepath, _save_dir)


if __name__ == '__main__':
    main()
    
    