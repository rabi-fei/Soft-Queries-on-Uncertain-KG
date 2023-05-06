import os

each_num = 20
for start in range(0, 600, each_num):
    command = ("nohup python data_preparation/sample_query.py "
               f"--start_index {start} "
               f"--end_index {start + each_num - 1} > sample_EFOX_{start}_{start + each_num}.log 2>&1 &")
    print(command, '\n')
    os.system(command)
    print("launched")
