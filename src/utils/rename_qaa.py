import json
import os.path as osp


from lifted_embedding_estimation_with_truth_value import DNF_lstr2name

false_name2lstr = {
    "1p": "r1(s1,f)",
    "2p": "r1(s1,e1)&r2(e1,f)",  # 2p
    "3p": "r1(s1,e1)&r2(e1,e2)&r3(e2,f)",  # 3p
    "2i": "r1(s1,f)&r2(s2,f)",  # 2i
    "3i": "r1(s1,f)&r2(s2,f)&r3(s3,f)",  # 3i
    "ip": "r1(s1,e1)&r2(s2,e1)&r3(e1,f)",  # ip
    "pi": "r1(s1,e1)&r2(e1,f)&r3(s2,f)",  # pi
    "2in": "r1(s1,f)&!r2(s2,f)",  # 2in
    "3in": "r1(s1,f)&r2(s2,f)&!r3(s3,f)",  # 3in
    "inp": "r1(s1,e1)&!r2(s2,e1)&r3(e1,f)",  # inp
    "pin": "r1(s1,e1)&r2(e1,f)&!r3(s2,f)",  # pin
    "pni": "r1(s1,e1)&!r2(e1,f)&r3(s2,f)",  # pni
    "2u": "r1(s1,f)|r2(s2,f)",  # 2u
    "up": "r1(s1,e1)|r2(s2,e1))&r3(e1,f)",  # up
}

if __name__ == "__main__":
    original_folder = 'FB15k-237-betae'
    output_folder = 'FB15k-237-EFO1'

    f_old = open(osp.join(original_folder, 'valid-qaa.json'))
    old_data = json.load(f_old)
    new_data = {}
    for DNF_lstr in DNF_lstr2name:
        beta_name = DNF_lstr2name[DNF_lstr]
        old_lstr = false_name2lstr[beta_name]
        new_data[DNF_lstr] = old_data[old_lstr]
    g = open(osp.join(output_folder, 'valid-qaa.json'))
    g.write(new_data)
