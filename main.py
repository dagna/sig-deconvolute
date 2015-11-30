import sys, getopt, nimfa, numpy.matlib
import multiprocessing as mp
import scipy.io, scipy.spatial
import sklearn.metrics
from numpy import *

WEAK_MUTATION_PERCENT = 0.01 # (removeWeak)
PERCENT_RECON_REMOVE = 0.07 # (filterOutIterations)
NUM_CORES = 4
NUM_SIGNATURES = 25 # same as rank
NUM_BOOTSTRAPS = 4 # normally 1000
ITERATIONS_PER_CORE = NUM_BOOTSTRAPS / NUM_CORES
BIG_NUMBER = 100 # assign as distance when a vector has been chosen for a cluster so doesn't get picked again (kmeans)
CONVERG_CUTOFF = 0.005 # considered a negligible change in cosine distance, used to declare convergence (kmeans)
CONVERG_ITER = 10 # num times accept negligible change before declaring convergence (kmeans)
TOTAL_REPLICATES = 100 # max times will assign new centroids (kmeans)
TOTAL_INIT_CONDITIONS = 5 # num times to run kmeans w different bootstraps as starting
DISTANCE_METRIC = 'cosine' # used to compare topics/signatures


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
    totalMutationsToRemove = sum(cumsum(sortedTotalsForEachMutation)/totalMutations < \
                                     WEAK_MUTATION_PERCENT)
    originalIndicesToRemove = originalSortedIndices[0: totalMutationsToRemove]

    #remove rows
    genomes = data
    genomes = delete(genomes, originalIndicesToRemove, 0)

    return originalIndicesToRemove

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

    removeIterationSets = removeIterationSets.astype(int)
    return removeIterationSets

def custKMeans(Wall, Hall, numberSignaturesToExtract, TOTAL_REPLICATES, distanceMetric, centroids, centroidsStd, exposure, exposureStd, idx, idxS, processStab, processStabAvg, clusterCompactness):

    #the topics within a single bootstrap must be assigned to different clusters
    def checkWall(Wall):
        file = scipy.io.loadmat("./tryit.mat")
        for i in range(size(Wall, 1)):
            if ~(Wall[:,i] == file['Wall'][:,i]).all():
                print 'mismatching column:{}'.format(str(i))

    minClusterDist = BIG_NUMBER # to be considered an acceptable cluster
    totalIter = size(Wall, 1) / numberSignaturesToExtract
    bootstrapIndices = numpy.array(range(0, size(Wall,1), numberSignaturesToExtract))
    randBootstrapIndices = bootstrapIndices[numpy.random.permutation(totalIter)]

    for iInitData in range(min(TOTAL_INIT_CONDITIONS, totalIter)):
        bootstrapIndexStart = randBootstrapIndices[iInitData]
        bootstrapIndexEnd = bootstrapIndexStart + numberSignaturesToExtract
        centroids = Wall[:, bootstrapIndexStart:bootstrapIndexEnd].copy() #otherwise will modify original Wall :(
        oldCentroids = numpy.random.rand(size(centroids,0), size(centroids,1))
        convergeCount = 0

        for iRep in range(TOTAL_REPLICATES):
            allDist = scipy.spatial.distance.squareform( scipy.spatial.distance.pdist( numpy.transpose(numpy.concatenate( (centroids, Wall) , axis=1)), distanceMetric ) )
            centroidDist = numpy.transpose(allDist[0:size(centroids, 1), size(centroids, 1) : size(allDist, 1)])

            jRange = numpy.random.permutation(numberSignaturesToExtract) #randomize the order in which clusters get to be assigned topics every replicate
            for jIndex in range(numberSignaturesToExtract):
                j = jRange[jIndex] #WHY DOES IT ASSIGN SAME NUMBER TWICE

                for i in range(0, size(Wall,1), numberSignaturesToExtract):
                    iRange = range(i, i + numberSignaturesToExtract)
                    Ind = numpy.argmin(centroidDist[iRange, j])
                    centroidDist[iRange[Ind], :] = BIG_NUMBER
                    idx[iRange[Ind]] = j


            maxDistToNewCentroids = 0
            for i in range(numberSignaturesToExtract):
                centroids[:, i] = numpy.mean(Wall[:, (idx == i).flatten()], axis=1) #calculate new centroids
                maxDistToNewCentroids = max(maxDistToNewCentroids, scipy.spatial.distance.pdist( numpy.transpose(numpy.concatenate( (centroids[:,i].reshape(size(centroids[:,i]), 1), oldCentroids[:, i].reshape(size(centroids[:,i]), 1)) , axis=1)), metric=distanceMetric ))

            if maxDistToNewCentroids < CONVERG_CUTOFF:
                convergeCount+= 1
            else:
                convergeCount = 0
                oldCentroids = centroids

            if convergeCount == CONVERG_ITER:
                break

        for i in range(numberSignaturesToExtract):
            clusterDist = scipy.spatial.distance.squareform( scipy.spatial.distance.pdist( numpy.transpose(numpy.concatenate( (centroids[:,i].reshape(size(centroids[:,i]), 1), Wall[:, (idx==i).flatten()]) , axis=1)), distanceMetric ) )
            clusterCompactness[i,:] = clusterDist[0, 1:size(clusterDist, 1)]

        if minClusterDist > mean(clusterCompactness[:]):
            minClusterDist = mean(clusterCompactness[:])
            centroidsFinal = centroids
            idxFinal = idx
            clusterCompactnessFinal = clusterCompactness

    centroids = numpy.transpose(centroidsFinal)
    idx = idxFinal
    clusterCompactness = clusterCompactnessFinal


    file = scipy.io.loadmat("./idxtest.mat")
    idx = file['idx']
    idx = idx - 1
    centroids = file['centroids']
    clusterCompactness = file['clusterCompactness']# put a breakpoint here
    centDist = mean(clusterCompactness, axis=1) #same

    # rearranging centroids with tightest clusters first
    centDistInd = numpy.argsort(centDist) #same
    clusterCompactness = clusterCompactness[centDistInd, :] #same
    centroids = centroids[centDistInd, :]
    idxNew = numpy.copy(idx)
    # change naming of indices so best cluster is 1
    for i in range(numberSignaturesToExtract):
        idxNew[(idx == centDistInd[i])] = i 
     #this doesnt do what you think it does.


    idx = idxNew



    if numberSignaturesToExtract > 1:
        processStab = sklearn.metrics.silhouette_samples(numpy.transpose(Wall), idx.ravel(), metric=DISTANCE_METRIC)
        for i in range(numberSignaturesToExtract):
            processStabAvg[0,i] = mean(processStab[(idx==i).ravel()])
    else:
        allDist = scipy.spatial.distance.squareform( scipy.spatial.distance.pdist( numpy.transpose(numpy.concatenate( (numpy.transpose(centroids), Wall) , axis=1)), distanceMetric ) )        
        processStab = 1 - numpy.transpose(allDist[0:size(numpy.transpose(centroids), 1), size(numpy.transpose(centroids), 1): size(allDist, 1) ])
        processStabAvg = mean(processStab)

    for i in range(numberSignaturesToExtract):
        centroidsStd[i,:] = std(Wall[:, (idx==i).flatten()], axis=1, ddof=1)

    centroids = numpy.transpose(centroids)
    centroidsStd = numpy.transpose(centroidsStd)

    # the indices i in idxS are assigned are assigned the indices of idx that were assigned to cluster i
    for i in range(0, size(Wall,2), numberSignaturesToExtract):
        iEnd = i + numberSignaturesToExtract
        idxG = idx[i:iEnd]

        for j in range(numberSignaturesToExtract):
            idxS[i+j,:] = numpy.nonzero(idxG == j)

    for i in range(numberSignaturesToExtract):
        exposure[i, :] = mean(Hall[idx==i, :])
        exposureStd[i, :] = std(Hall[(idx==i).flatten(),:], axis=0, ddof=1)

# Add zeros at indices that weak mutations were removed at previously
def addWeak(mutationTypesToAddSet, processes, processesStd, Wall, genomeErrors, genomesReconstructed):

    totalMutTypes = size(Wall, 0) + size(mutationTypesToAddSet)

    origArrayIndex = 1
    for i in range(totalMutTypes):
        if ~any(mutationTypesToAddSet==i):
            break



























#take out the labels that came with the dataset
inputfile = fetch_arg(sys.argv[1:])
data = loadtxt(strip_first_col(inputfile), skiprows=1);
indicesToRemove = removeWeak(data)
data = numpy.delete(data, indicesToRemove, 0)
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

removeIterationSets = filterOutIterations(Wall, Hall, genomeErrors, NUM_SIGNATURES, genomesReconstructed, PERCENT_RECON_REMOVE)
Wall = numpy.delete(Wall, removeIterationSets, 1)
Hall = numpy.delete(Hall, removeIterationSets, 0)
genomeErrors = numpy.delete(genomeErrors, removeIterationSets, 2)
genomesReconstructed = numpy.delete(genomesReconstructed, removeIterationSets, 2)

file = scipy.io.loadmat("/Users/Andromeda/Desktop/sig-deconvolute/filtered.mat")
Wall = file['Wall']
Hall = file['Hall']
genomeErrors = file['genomeErrors']
genomesReconstructed = file['genomesReconstructed'] 




centroids = numpy.zeros((size(Wall,0), NUM_SIGNATURES)) 
centroidsStd = numpy.zeros((size(centroids))) # will later represent clustered signatures
exposure = numpy.zeros((NUM_SIGNATURES,size(Hall,1)))
exposureStd = numpy.zeros((NUM_SIGNATURES,size(Hall,1)))
clusterCompactness = numpy.zeros((NUM_SIGNATURES, size(Wall, 1) / NUM_SIGNATURES))
idx = numpy.zeros(shape=(size(Hall, 0), 1))
idxS = numpy.zeros(shape=(size(Hall, 0), 1))
processStab = numpy.zeros(shape=(size(Wall,1)))
processStabAvg = numpy.zeros(shape=(1, NUM_SIGNATURES))

custKMeans(Wall, Hall, NUM_SIGNATURES, TOTAL_REPLICATES, DISTANCE_METRIC, centroids, centroidsStd, exposure, exposureStd, idx, idxS, processStab, processStabAvg, clusterCompactness)




# extract(data, 1, 25, w, h)


