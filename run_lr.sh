python3 task_dependent_train.py --learning_rate=1e-1 --output_dir log/learning_rate_1e-1 --device cuda:0 &
python3 task_dependent_train.py --learning_rate=1e-2 --output_dir log/learning_rate_1e-2 --device cuda:1 &
python3 task_dependent_train.py --learning_rate=1e-3 --output_dir log/learning_rate_1e-3 --device cuda:2 &
python3 task_dependent_train.py --learning_rate=1e-4 --output_dir log/learning_rate_1e-4 --device cuda:3