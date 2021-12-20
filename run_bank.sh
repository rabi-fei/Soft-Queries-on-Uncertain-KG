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