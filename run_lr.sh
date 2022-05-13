python3 task_dependent_train.py --objective noisy --batch_size=512  --learning_rate=1e-3 --output_dir log/noisy_lr_1e-3_bs_512  --device cuda:0 &
python3 task_dependent_train.py --objective noisy --batch_size=512  --learning_rate=1e-4 --output_dir log/noisy_lr_1e-4_bs_512  --device cuda:1 &
python3 task_dependent_train.py --objective noisy --batch_size=1024 --learning_rate=1e-3 --output_dir log/noisy_lr_1e-3_bs_1024 --device cuda:2 &
python3 task_dependent_train.py --objective noisy --batch_size=1024 --learning_rate=1e-4 --output_dir log/noisy_lr_1e-4_bs_1024 --device cuda:3
