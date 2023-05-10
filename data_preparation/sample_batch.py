import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=150)
parser.add_argument("--dataset", type=str, default='NELL')
parser.add_argument("--each", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    each_num = args.each
    for start in range(args.start, args.end, each_num):
        command = ("nohup python data_preparation/sample_query.py "
                   f"--start_index {start} "
                   f"--end_index {start + each_num - 1} > sample_EFOX_{dataset}_{start}_{start + each_num - 1}.log 2>&1 &")
        print(command, '\n')
        os.system(command)
        print("launched")

