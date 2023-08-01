# BinMoCo
Source code of BinMoCo.

## Environment
```
radare2 5.7.8 + r2pipe 1.7.4
torch-geometric 2.1.0
transformers 4.28.1
pytorch 1.12.1
python 3.10.6
lightning 2.0.2
connectorx 0.3.1
faiss-gpu 1.7.2
```

## Train and Evaluate on BFS dataset

0. Download the BFS dataset [here]((https://drive.google.com/drive/folders/1uqZb0geb4CgDe9XEczZhNcyfBQM1TusG)), which is proposed in paper *How Machine Learning Is Solving the Binary Function Similarity Problem*


1. Build BFS Database

```
python code/build_db.py --db_name sec22
```

- This will generate `db.sqlite` in `database/sec22`

2. Build function groups

```
python code/build_group.py database/sec22
```

3. Build vocabs

```
python code/build_vocab.py database/sec22
python code/build_imp_vocab.py database/sec22
```

4. Train BinMoCo

```
python code/train.py \
    --data_dir database/sec22 \
    --data_repr cfg_cg \
    --num_edge_type 3 \
    --embedding_dims 128 \
    --seq_model transformer \
    --seq_hidden_dims 128 \
    --seq_layers 4 \
    --gnn_name gatedgcn \
    --gnn_hidden_dims 128 \
    --gnn_layers 5 \
    --gnn_out_dims 128 \
    --train_batch_size 30 \
    --batch_k 5 \
    --val_batch_size 30 \
    --train_num_each_epoch 300000 \
    --num_epochs 30 \
    --num_workers 12 \
    --learning_rate 0.001 \
    --miner_type ms \
    --loss_type ms \
    --use_moco \
    --memory_size 16384 \
    --early_stopping 15 \
    --precision 16 \
    --save_name sec22_cfg_cg_trans_gatedgcn_gin_ms_ms_moco
```

5. Build test data (four tasks: XO, XC, XA, XM, with different Poolsizes)

```
python code/build_testdata.py database/sec22
```

6. Test the trained model

```
python code/test.py results/dml/sec22_cfg_cg_trans_gatedgcn_gin_ms_ms_moco/version_0
```


## Reference
We refer to the following repositories during implementation:

- SAFE: [source code](https://github.com/gadiluna/SAFE)
- jTrans: [source code](https://github.com/vul337/jTrans)
- my solution for [CCF BDCI 2022 - Linux跨平台二进制函数识别](https://datafountain.cn/competitions/593): [source code](https://github.com/Bowen-n/bcsd_ms)

