from numpy import *
import sys, getopt, os, numpy.matlib

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

#call once
def removeWeak(data):
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

    return genomes
    
def bootstrapCancerGenomes(genomes):

    #the probability dist to pull from
    sumGenomes = sum(genomes, 0)
    normGenomes = genomes / numpy.matlib.repmat(sumGenomes, size(genomes,0), 1)
    
    #treating every genome as pulling from a multinommial distribution with K outcomes.
    ans = numpy.empty(shape=(size(genomes,0), size(genomes,1)))
    for i in range(size(genomes,1)):
        ans[:,i] = numpy.random.multinomial(sumGenomes[i],normGenomes[:,i])

    return ans

#run this from from z=1 to min (K,G) for edited genomes, and 1000 iterations for each z. iterations per core = numcores/1000  
def extract(genomes, totalIterationsPerCore, numberSignaturesToExtract):
    for i in range(totalIterationsPerCore):
        print(i)










#take out the labels that came with the dataset
inputfile = fetch_arg(sys.argv[1:])
data = loadtxt(strip_first_col(inputfile), skiprows=1);
bootstrap = bootstrapCancerGenomes(data)


