import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=150)
parser.add_argument("--dataset", type=str, default='FB15k-237')
parser.add_argument("--each", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    datafolder = 'data/' + dataset + '-EFO1'
    output_folder = 'data/' + dataset + '-EFOX'
    each_num = args.each
    for start in range(args.start, args.end, each_num):
        command = ("nohup python data_preparation/sample_query.py "
                   f"--data_folder {datafolder} "
                   f"--output_folder {output_folder} "
                   f"--start_index {start} "
                   f"--end_index {start + each_num - 1} > sample_EFOX_{dataset}_{start}_{start + each_num - 1}.log 2>&1 &")
        print(command, '\n')
        os.system(command)
        print("launched")

