import sys, getopt, nimfa, numpy.matlib
import multiprocessing as mp
from numpy import *

WEAK_MUTATION_PERCENT = 0.01
PERCENT_RECON_REMOVE = 0.07
NUM_CORES = 4
NUM_SIGNATURES = 25 # same as rank
NUM_BOOTSTRAPS = 4 # normally 1000
ITERATIONS_PER_CORE = NUM_BOOTSTRAPS / NUM_CORES

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

#run this from from z=1 to min (K,G) for edited genomes, and 1000 iterations for each z. iterations per core = 1000/numcores
def extract(genomes, totalIterationsPerCore, numberSignaturesToExtract, WPerCore, HPerCore, genomeErrorsPerCore, genomesReconstructedPerCore ):

    totalMutationTypes = size(data,0)
    totalGenomes = size(data, 1)
    processCount = 0

    for i in range(totalIterationsPerCore):
        #replacing zeroes w small number

        bootstrapGenomes = numpy.maximum(bootstrapCancerGenomes(genomes), numpy.finfo(numpy.float32).eps)
        nmf = nimfa.Nmf(bootstrapGenomes, max_iter=3, rank=numberSignaturesToExtract, update='divergence', objective='conn', conn_change=10000, test_conv=10,) # max iter is actual 1 mill
        nmf_fit = nmf()
        for j in range(numberSignaturesToExtract):
        
            total = sum(nmf_fit.basis()[:,j])
            nmf_fit.basis()[:,j] = nmf_fit.basis()[:,j] / total
            nmf_fit.coef()[j,:] = nmf_fit.coef()[j,:] / total

        genomeErrorsPerCore[:, :, i] = bootstrapGenomes - nmf_fit.basis() * nmf_fit.coef()
        genomesReconstructedPerCore[:, :, i] = nmf_fit.basis() * nmf_fit.coef()
        WPerCore[:, processCount : (processCount + numberSignaturesToExtract)] = nmf_fit.basis()
        HPerCore[processCount : (processCount + numberSignaturesToExtract), :] = nmf_fit.coef()
        processCount = processCount + numberSignaturesToExtract


def filterOutIterations(Wall, Hall, genomeErrors, numberSignaturesToExtract, genomesReconstructed, removePercentage):

    totalIterations = size(Wall, 1) / numberSignaturesToExtract
    totalRemoveIter = int(round(removePercentage * totalIterations))

    closenessGenomes = numpy.zeros(shape=(totalIterations, 1))
    for i in range(totalIterations):
        closenessGenomes[i] = numpy.linalg.norm(genomeErrors[:, :, i])

    index = numpy.argsort(closenessGenomes, 0)[::-1]
    removeIterations = index[0:totalRemoveIter]

    removeIterationSets = numpy.zeros(shape=(numberSignaturesToExtract * totalRemoveIter, 1))
    

    for i in range(totalRemoveIter):
        iStart = numberSignaturesToExtract * removeIterations[i]
        iEnd = numberSignaturesToExtract * (removeIterations[i] + 1)

        removeIterationSets[numberSignaturesToExtract*i : numberSignaturesToExtract*(i+1), :] = numpy.arange(iStart,iEnd).reshape(len(numpy.arange(iStart,iEnd)), 1)

    return removeIterationSets








#take out the labels that came with the dataset
inputfile = fetch_arg(sys.argv[1:])
data = loadtxt(strip_first_col(inputfile), skiprows=1);
#data = removeWeak(data)
totalMutationTypes = size(data,0)
totalGenomes = size(data, 1)

Wall = numpy.zeros(shape=(totalMutationTypes, NUM_SIGNATURES * ITERATIONS_PER_CORE * NUM_CORES))
Hall = numpy.zeros(shape=( NUM_SIGNATURES * ITERATIONS_PER_CORE * NUM_CORES , totalGenomes))
genomeErrors = numpy.zeros(shape= (totalMutationTypes, totalGenomes, ITERATIONS_PER_CORE * NUM_CORES))
genomesReconstructed = numpy.zeros(shape= (totalMutationTypes, totalGenomes, ITERATIONS_PER_CORE * NUM_CORES))



for i in range(NUM_CORES): #smpd statement in matlab
    WPerCore = numpy.zeros(shape=(totalMutationTypes, NUM_SIGNATURES * ITERATIONS_PER_CORE ))
    HPerCore = numpy.zeros(shape=( NUM_SIGNATURES * ITERATIONS_PER_CORE , totalGenomes))
    # every iteration has an error in its reconstruction (which is a matrix)
    genomeErrorsPerCore = numpy.zeros(shape= (totalMutationTypes, totalGenomes, ITERATIONS_PER_CORE))
    genomesReconstructedPerCore = numpy.zeros(shape= (totalMutationTypes, totalGenomes, ITERATIONS_PER_CORE))

    extract(data, ITERATIONS_PER_CORE, NUM_SIGNATURES, WPerCore, HPerCore, genomeErrorsPerCore, genomesReconstructedPerCore)

    stepAll = NUM_SIGNATURES * ITERATIONS_PER_CORE
    startAll = stepAll * i
    endAll = startAll + stepAll
    Wall[:, startAll:endAll] = WPerCore
    Hall[startAll:endAll, :] = HPerCore

    stepAll = ITERATIONS_PER_CORE
    startAll = stepAll * i
    endAll = startAll + stepAll
    genomeErrors[:, :, startAll:endAll] = genomeErrorsPerCore
    genomesReconstructed[:, :, startAll:endAll] = genomesReconstructedPerCore



# extract(data, 1, 25, w, h)


