import os

each_num = 5
dataset = 'NELL'
for start in range(0, 150, each_num):
    command = ("nohup python data_preparation/sample_query.py "
               f"--start_index {start} "
               f"--end_index {start + each_num - 1} > sample_EFOX_{dataset}_{start}_{start + each_num - 1}.log 2>&1 &")
    print(command, '\n')
    os.system(command)
    print("launched")
