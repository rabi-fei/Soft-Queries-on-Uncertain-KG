import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=150)
parser.add_argument("--end", type=int, default=250)
parser.add_argument("--dataset", type=str, default='FB15k-237')
parser.add_argument("--each", type=int, default=5)
parser.add_argument("--sample", type=int, default='1000')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    datafolder = 'data/' + dataset + '-EFOX'
    output_folder = 'data/' + dataset + '-EFOX'
    each_num = args.each
    sample_num = args.sample
    if dataset == 'NELL':
        sample_num = int(sample_num * 0.6)
    for start in range(args.start, args.end, each_num):
        command = ("nohup python sample_query.py "
                   f"--data_folder {datafolder} "
                   f"--num_samples {sample_num} "
                   f"--output_folder {output_folder} "
                   f"--start_index {start} "
                   f"--end_index {start + each_num - 1} > sample_EFOX_{dataset}_{start}_{start + each_num - 1}.log 2>&1 &")
        print(command, '\n')
        os.system(command)
        print("launched")

