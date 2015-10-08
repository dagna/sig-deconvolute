from numpy import *
import sys, getopt

def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
               yield line.split(delimiter, 1)[1]
            except IndexError:
               continue

def fetch_arg(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"i:",["ifile="])
    except getopt.GetoptError:
        print('bad')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            inputfile = arg
    
    return inputfile
    
    
    
#def extract(genomes, totalIterationsPerCOre, numberSignaturesToExtract)
#take out the labels that came with the dataset
inputfile = fetch_arg(sys.argv[1:])
data = loadtxt(strip_first_col(inputfile), skiprows=1);

