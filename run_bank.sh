python3 train.py --log_dir log/efl_round5_family --num_steps 50000 --eval_every 5000

python3 train.py --log_dir log/fb15k/efl_round3 \
                 --learning_method efl --efl_round 5 \
                 --num_steps 50000 \
                 --eval_every 5000 \
                 --lr 1e-2 \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/FB15K/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/FB15K/edges_as_id_valid.tsv \
                 --auto_index False \
                 --cuda 1

python3 train.py --log_dir log/fb15k/lpl \
                 --learning_method lpl \
                 --num_steps 50000 \
                 --eval_every 5000 \
                 --lr 1e-2 \
                 --batch_size 512 \
                 --train_data datasets-knowledge-embedding/FB15K/edges_as_id_train.tsv \
                 --dev_data datasets-knowledge-embedding/FB15K/edges_as_id_valid.tsv \
                 --auto_index False \
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
python train.py --learning_method=lpl --log_dir log/family/lpl --eval_every 5000 --batch_size 512 --cuda 1