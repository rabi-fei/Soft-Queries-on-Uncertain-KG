import sys
import os
sys.path.append(os.getcwd())

from language.fof import sample_efo_dnf_query

if __name__ == "__main__":
    sample_efo_dnf_query(p=3, q=3, r=3, per=0.5, l=5, k=3, pn=0.5)