python3 train.py --log_dir log/efl_round5_family --num_steps 50000 --eval_every 5000

python3 train.py --log_dir log/fb15k/efl_round5 \
                 --learning_method efl \
                 --lr 1e-2 \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/FB15K/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/FB15K/edges_as_id_valid.tsv \
                 --test_data datasets-knowledge-embedding/FB15K/edges_as_id_test.tsv \
                 --cuda 1

python3 train.py --log_dir log/fb15k/lpl-transe \
                 --learning_method lpl \
                 --num_steps 50000 \
                 --eval_every 5000 \
                 --lr 1e-2 \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/FB15K/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/FB15K/edges_as_id_valid.tsv \
                 --cuda 2

# good learning rate
python3 train.py --log_dir log/efl_round5_family_lr1e-2 --num_steps 50000 --eval_every 5000 --lr 1e-2

# good learning rate, efl round = 3
python3 train.py --log_dir log/family/efl_round3_lr1e-2 --num_steps 50000 --eval_every 5000 --lr 1e-2 --efl_round 3

python3 train.py --log_dir log/family/efl_round3_lr1e-2_batch_size=256 --num_steps 50000 --eval_every 5000 --lr 1e-2 --efl_round 3 --batch_size 256 --cuda 2

python3 train.py --log_dir log/family/efl_round3_lr1e-2_batch_size=512 --num_steps 50000 --eval_every 5000 --lr 1e-2 --efl_round 3 --batch_size 512 --cuda 2

# good learning rate, efl round = 1
python3 train.py --log_dir log/efl_round1_family_lr1e-2 --num_steps 50000 --eval_every 5000 --lr 1e-2 --efl_round 1

# justify the batch size issue
python3 train.py --log_dir log/efl_round1_family_lr1e-2_batch_size=512 --num_steps 50000 --eval_every 5000 --lr 1e-2 --efl_round 1 --batch_size 512

# LPL family
python train.py --learning_method=lpl --log_dir log/family/lpl --batch_size 512 --cuda 1


python3 train.py --log_dir log/WN18RR/efl_round5 \
                 --learning_method efl \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/WN18RR/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/WN18RR/edges_as_id_valid.tsv \
                 --test_data datasets-knowledge-embedding/WN18RR/edges_as_id_test.tsv \
                 --cuda 3

python3 train.py --log_dir log/WN18RR/lpl-transe \
                 --learning_method lpl \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/WN18RR/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/WN18RR/edges_as_id_valid.tsv \
                 --test_data datasets-knowledge-embedding/WN18RR/edges_as_id_test.tsv \
                 --cuda 3


python3 tast_dependent_train.py --learning_rate=1e-1 --output_dir log/learning_rate_1e-1 --device cuda:0
python3 tast_dependent_train.py --learning_rate=1e-2 --output_dir log/learning_rate_1e-2 --device cuda:1
python3 tast_dependent_train.py --learning_rate=1e-3 --output_dir log/learning_rate_1e-3 --device cuda:2
python3 tast_dependent_train.py --learning_rate=1e-4 --output_dir log/learning_rate_1e-4 --device cuda:3


python3 task_dependent_train.py --device cuda:0 --reasoning_rate 0.1 --embedding_dim 1000 --checkpoint_path pretrain/complex/FB15k-237-model-rank-1000-epoch-100-1602508358.pt


task_dependent_train.py --embedding_dim 100 --device cuda:1 --checkpoint_path pretrain/complex/FB15k-237-model-rank-100-epoch-100-1602503352.pt --sigma 1 --learning_rate 1e-3 --batch_size 1024 --reasoning_steps 10 --objective noisy --output_dir log/q2b


task_dependent_train.py --embedding_dim 100 --device cuda:2 --checkpoint_path pretrain/complex/FB15k-237-model-rank-100-epoch-100-1602503352.pt --sigma 1 --learning_rate 1e-2 --batch_size 32 --reasoning_steps 10 --objective none --output_dir log/test_plus_eval --model_name complexplus


python3 task_dependent_train.py --embedding_dim 100 --device cuda:0 --checkpoint_path pretrain/complex/FB15k-237-model-rank-100-epoch-100-1602503352.pt --sigma 1 --learning_rate 1e-2 --batch_size 32 --reasoning_steps 10 --objective noisy --output_dir log/test_plus --model_name complexplus


python3 task_dependent_train.py --embedding_dim 1000 --device cuda:0 --checkpoint_path pretrain/complex/FB15k-237-model-rank-1000-epoch-100-1602508358.pt --learning_rate 1e-4 --batch_size 32 --reasoning_steps 10 --objective noisy --output_dir log/train_query_train_1p_eval1kRsteps