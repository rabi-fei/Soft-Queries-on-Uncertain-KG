
python3 task_dependent_train.py --objective kvsall --learning_rate=1e-3 --output_dir log/kvsall_lr_1e-3 --device cuda:1 &
python3 task_dependent_train.py --objective kvsall --learning_rate=1e-4 --output_dir log/kvsall_lr_1e-4 --device cuda:2 &

python3 task_dependent_train.py --objective noisy  --learning_rate=1e-3 --output_dir log/noisy_lr_1e-3 --device cuda:3 &
python3 task_dependent_train.py --objective noisy  --learning_rate=1e-4 --output_dir log/noisy_lr_1e-4 --device cuda:4