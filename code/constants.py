import os

root_dir = os.path.realpath(os.path.join(__file__, '..', '..'))
db_dir = os.path.join(root_dir, 'database')
log_dir = os.path.join(root_dir, 'logs')

os.makedirs(db_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

seed = 42

max_num_block = 128 # >= 97% of binkit_2.0 and sec22
max_block_token_len = 80 # >= 94% of binkit_2.0, >= 92% of sec22

max_ins_token_len = 8 # 99% of binkit dataset
max_block_ins_len = 40 # 90% of binkit blocks have instruction <= 14

# for pre-filter
min_num_block = 5
min_variant_num = 5

# CG degree
max_cg_in_degree = 20
max_cg_out_degree = 20

# * BinKit_v1 constants
# compiler: gcc-6.4.0 clang-6.0
# optimization: O0 O1 O2 O3 Os
# architecture: arm_32 arm_64 x86_32 x86_64 mips_32 mips_64 mipseb_32 mipseb_64

compilers = ['gcc-6.4.0', 'clang-6.0']
optimizations = ['O0', 'O1', 'O2', 'O3', 'Os']
architectures = ['arm', 'x86', 'mips', 'mipseb']
bits = ['32', '64']

binkit_train_projs = [
    'inetutils-1.9.4', 'glpk-4.65', 'dap-3.10', 'libtasn1-4.13', 'recutils-1.7', 
    'wdiff-1.2.2', 'sharutils-4.15.2', 'gcal-4.1', 'enscript-1.6.6', 'tar-1.30', 
    'gnu-pw-mgr-2.3.1', 'binutils-2.30', 'gdbm-1.15', 'osip-5.0.0', 'nettle-3.4', 
    'which-2.21', 'gsl-2.5', 'direvent-5.1', 'gnudos-1.11.4', 'macchanger-1.6.0', 
    'hello-2.10', 'xorriso-1.4.8', 'spell-1.1', 'gsasl-1.8.0', 'sed-4.5']

binkit_val_projs = [
    'libiconv-1.15', 'plotutils-2.6', 'coreutils-8.29', 'time-1.9', 'cppi-1.18', 
    'ccd2cue-0.5', 'patch-2.7.6'
]

binkit_test_projs = [
    'gawk-4.2.1', 'a2ps-4.14', 'readline-7.0', 'libunistring-0.9.10', 'gzip-1.9',
    'libtool-2.4.6', 'lightning-2.1.2', 'datamash-1.3', 'grep-3.1', 'libmicrohttpd-0.9.59',
    'findutils-4.6.0', 'cflow-1.5', 'texinfo-6.5', 'gmp-6.1.2', 'libidn-2.0.5',
    'bool-0.2.2', 'gss-1.0.3', 'units-2.16', 'cpio-2.12'
]

# * BinKit_v2 constants
# compiler: gcc-9.4.0 clang-9.0
# optimization: O0 O1 O2 O3 Os
# architecture: arm_32 arm_64 x86_32 x86_64 mips_32 mips_64

binkit_v2_compilers = ['gcc-9.4.0', 'clang-9.0']
binkit_v2_optimizations = ['O0', 'O1', 'O2', 'O3', 'Os']
binkit_v2_architectures = ['arm', 'x86', 'mips']
binkit_v2_bits = ['32', '64']

binkit_v2_train_projs = [
    'binutils-2.40', 'dap-3.10', 'direvent-5.3', 'enscript-1.6.6', 'gcal-4.1', 
    'gdbm-1.23', 'glpk-5.0', 'gnu-pw-mgr-2.7.4', 'gnudos-1.11.4', 'gsl-2.7.1', 
    'inetutils-2.4', 'libtasn1-4.19.0', 'macchanger-1.6.0', 'nettle-3.8.1', 'osip-5.3.1', 
    'recutils-1.9', 'sharutils-4.15.2', 'wdiff-1.2.2', 'which-2.21', 'xorriso-1.5.4'
]

binkit_v2_val_projs = [
    'ccd2cue-0.5', 'coreutils-9.1', 'cppi-1.18', 'libiconv-1.17', 'patch-2.7.5',
    'time-1.9', 'spell-1.1', 'hello-2.12.1', 'gsasl-2.2.0'
]

binkit_v2_test_projs = [
    'a2ps-4.14', 'cflow-1.7', 'cpio-2.12', 'datamash-1.8', 'findutils-4.9.0',
    'gawk-5.2.1', 'gmp-6.2.1', 'grep-3.8', 'gss-1.0.4', 'gzip-1.12',
    'libidn-2.3.4', 'libmicrohttpd-0.9.75', 'libtool-2.4.7', 'libunistring-1.1', 'lightning-2.2.0',
    'readline-8.2', 'texinfo-7.0.1', 'units-2.22', 'bool-0.2.2', 'tar-1.34',
    'sed-4.9'
]

# * Sec22 constants
# compiler: gcc-9 clang-9
# optimization: O0 O1 O2 O3 Os
# architecture: arm_32 arm_64 x86_32 x86_64 mips_32 mips_64
sec22_compilers = ['gcc-9', 'clang-9']
sec22_optimizations = ['O0', 'O1', 'O2', 'O3', 'Os']
sec22_architectures = ['arm', 'x86', 'mips']
sec22_bits = ['32', '64']

# {'unrar': 18720, 'zlib': 6365, 'curl': 37610, 'nmap': 95708, 'openssl': 222102, 'clamav': 169253, 'z3': 843106}
sec22_train_projs = ['z3', 'zlib', 'unrar']
sec22_val_projs = ['curl']
sec22_test_projs = ['openssl', 'clamav', 'nmap']


# * summary constants
compile_config = {
    'binkit': {
        'compiler': compilers,
        'opti': optimizations,
        'arch': architectures,
        'bit': bits
    }, 
    'binkit_2.0': {
        'compiler': binkit_v2_compilers,
        'opti': binkit_v2_optimizations,
        'arch': binkit_v2_architectures,
        'bit': binkit_v2_bits
    },
    'sec22': {
        'compiler': sec22_compilers,
        'opti': sec22_optimizations,
        'arch': sec22_architectures,
        'bit': sec22_bits
    }
}


# * vul detect constants
# MDC2_Update is not in TP-Link Deco-M4, others exist in all binaries
vul_funcnames = {
    'netgear': [
        "CMS_decrypt", # sym.CMS_decrypt
        "PKCS7_dataDecode", # sym.PKCS7_dataDecode
        "MDC2_Update", # sym.MDC2_Update
        "BN_bn2dec", # sym.BN_bn2dec
    ],
    'tp-link': [
        "X509_NAME_oneline", # sym.X509_NAME_oneline
        "EVP_EncryptUpdate", # sym.EVP_EncryptUpdate
        "EVP_EncodeUpdate", # sym.EVP_EncodeUpdate
        "SRP_VBASE_get_by_user", # sym.SRP_VBASE_get_by_user
        "BN_dec2bn", # sym.BN_dec2bn
        "BN_hex2bn" # sym.BN_hex2bn
    ]
}

# 10 functions cover 8 CVEs
cve_dict = {
    "CMS_decrypt": "CVE-2019-1563",
    "PKCS7_dataDecode": "CVE-2019-1563",
    "MDC2_Update": "CVE-2016-6303",
    "BN_bn2dec": "CVE-2016-2182",
    "X509_NAME_oneline": "CVE-2016-2176",
    "EVP_EncryptUpdate": "CVE-2016-2106",
    "EVP_EncodeUpdate": "CVE-2016-2105",
    "SRP_VBASE_get_by_user": "CVE-2016-0798",
    "BN_dec2bn": "CVE-2016-0797",
    "BN_hex2bn": "CVE-2016-0797"
}


if __name__ == '__main__':
    import json
    
    # binkit_dir = 'database/binkit'
    # with open(os.path.join(binkit_dir, 'split.json'), 'w') as f:
    #     json.dump({
    #         'train': binkit_train_projs,
    #         'val': binkit_val_projs,
    #         'test': binkit_test_projs
    #     }, f, indent=1)

    binkit_v2_dir = 'database/binkit_2.0'
    with open(os.path.join(binkit_v2_dir, 'split.json'), 'w') as f:
        json.dump({
            'train': binkit_v2_train_projs,
            'val': binkit_v2_val_projs,
            'test': binkit_v2_test_projs
        }, f, indent=1)
    
    sec22_dir = 'database/sec22'
    with open(os.path.join(sec22_dir, 'split.json'), 'w') as f:
        json.dump({
            'train': sec22_train_projs,
            'val': sec22_val_projs,
            'test': sec22_test_projs
        }, f, indent=1)
        
