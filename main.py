from numpy import *
import sys, getopt, os

WEAK_MUTATION_PERCENT = 0.01
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

#removing the mutations that cumulatively account for less than 1%
sortedTotalsForEachMutation = sort(sum(data, 1))

originalSortedIndices = argsort(sum(data,1))
totalMutations = sum(data)
totalMutationsToRemove = sum(cumsum(sortedTotalsForEachMutation)/ totalMutations < \
                                 WEAK_MUTATION_PERCENT)
originalIndicesToRemove = originalSortedIndices[0: totalMutationsToRemove]

#remove rows
genomes = data
genomes = delete(genomes, originalIndicesToRemove, 0)


