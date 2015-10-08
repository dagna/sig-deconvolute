from numpy import *

def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue

data = loadtxt(strip_first_col('Breast_genomes_mutational_catalog_96_subs.txt'), skiprows=1);