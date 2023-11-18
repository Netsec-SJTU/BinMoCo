import argparse
import os

from constants import db_dir
from dataset.factory import BinKit_v2Factory, Sec22Factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, choices=('sec22', 'binkit_2.0'))
    args = parser.parse_args()

    if args.db_name == 'sec22':
        factory = Sec22Factory(os.path.join(db_dir, 'sec22', 'db.sqlite'))
        factory.build_db(
            ['/home/user/data/bcsd/security22_review/Dataset-1'], 
            num_proc=6
        )

    elif args.db_name == 'binkit_2.0':
        factory = BinKit_v2Factory(os.path.join(db_dir, 'binkit_2.0', 'db.sqlite'))
        factory.build_db(
            ['/home/user/data/bcsd/binkit_2.0/BinKit_normal'],
            num_proc=12
        )


if __name__ == '__main__':
    main()

