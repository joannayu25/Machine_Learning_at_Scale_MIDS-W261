#!/usr/bin/env python


import re
import ast
import time
import numpy as np
import pandas as pd
import pyspark
from pyspark.accumulators import AccumulatorParam

conf = pyspark.SparkConf().setAll([ ('spark.executor.pyspark.memory', '11g'), ('spark.driver.memory','11g')])
sc = pyspark.SparkContext(conf=conf)

#sc = pyspark.SparkContext()


############## YOUR BUCKET HERE ###############

BUCKET="jyu-mids-w261-2019fall"

############## (END) YOUR BUCKET ###############


wikiRDD = sc.textFile("gs://"+BUCKET+"/wiki_graph.txt")


def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############
    
    # write any helper functions here
    
    # Tokenize each line and emit a pair RDD in the 
    # (node_id , edges) format for all nodes
    # including: 1) nodes with real edges, 2) dangling nodes
    # where the 'edges' list is set to the empty dict {}. 
    def emit_nodes(line):
        node, edges = line.split('\t')
        edge_nodes = ast.literal_eval(edges)
        all_edges = ''
        for edge_node in edge_nodes.keys():
            weight=edge_nodes[edge_node]
           # all_edges=all_edges+','+edge_node+','*(weight-1)+edge_node*(weight-1)
            yield (edge_node, '')
            yield (node, edge_node+(','+edge_node)*(weight-1) )
      #  yield (node, all_edges)

    def combineEdges(x, y):
        if x=='': return y
        elif y=='': return x
        else: return x+','+y

    # write your main Spark code here
    
    # For all nodes in the data, emit (node_id , edges).
    # Use 'reduceByKey' to take out duplicates and cache 
    # this 'tempRDD'.
    tempRDD = dataRDD.flatMap(emit_nodes) \
                     .reduceByKey(combineEdges) \
                     .cache()
    
    # Compute N by calling count() on 'tempRDD' and broadcast 
    # the value so the mappers can access it.
    totalCount = tempRDD.count()    
    init_value = sc.broadcast(1/totalCount)
    
    # Initalize the correct score (1/N) using the newly computed N. 
    graphRDD = tempRDD.map(lambda line: (line[0], (init_value.value, line[1])))
   
    ############## (END) YOUR CODE ###############
    
    return graphRDD

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2
    
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
   
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.

    # Mapper function to emit probability masses for each node's 
    # neighbors. 
    def emit_prob_mass(line):
        # Parse the line, tokenize each element, and emit the 
        # original record with 0 rank, (node_id, (0, {neighbors})),
        # to preserve the graph structure for the next iteration.
        node, (prob, edges) = line        
        yield node, (0, edges)        
        # For each neighboring node, calculate the probability mass 
        # being redistributed and emit in the format of (node_id, (score, {})) 
        # for each neighbor. Initially, I emitted one record per 
        # edge, so the repeated edges to the same neighbor were emitted 
        # multiple times. That would flood the stream but could leverage 
        # Spark's reduceByKey. The implementation below combines the 
        # repeated edges so fewer records are emitted to the stream, 
        # but more computation is done in this function. Both implementation 
        # were tested and combining repeated edges before emitting is slightly 
        # faster.
        if edges != '': 
            edge_list = edges.split(',')
            total_wt = len(edge_list)
            current = edge_list[0]
            weight=0
            # Iterate through the list of edges and emit the combined
            # probability once a new neighbor is encountered.  
            for i in range(total_wt):
                if edge_list[i]==current: 
                    weight+=1
                else:        
                    yield current, (prob/total_wt*weight, '')
                    weight=1
                    current=edge_list[i]
            if weight!=0: yield current, (prob/total_wt*weight, '')
            
    # Increment the accumulators (mmAccum and totAccum).
    def inc_accum(line):
        node, (rank, edges) = line
        totAccum.add(rank)
        if edges == '': 
            mmAccum.add(rank)
        
    # Computes the new rank by applying the formula.
    def update_rank(line):
        node, (rank, edges) = line
        new_rank = teleport_val.value + d.value*(mmass_per_node.value+rank)
        yield node, (new_rank, edges)        
        
    # Checks convergence given a threshold value broadcasted by the driver.
    # This function is not used since it is not required. 
    def not_converged(RDD_A, RDD_B):
        if RDD_A == None or RDD_B == None: 
            return True
        else: 
            for key in RDD_A.keys().collect():
                if np.absolute(RDD_A.lookup(key)[0][0]-RDD_B.lookup(key)[0][0]) > threshold.value:
                    return True
            return False
       
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace

    # Initialize variables before the while loop. 'Iteration' is to keep track 
    # of the number of iterations so far. 'threshold' is the difference between 
    # the probabilities that defines convergence, but it was not used.

    steadyStateRDD = graphInitRDD
    iteration = 0
    threshold = sc.broadcast(1e-10) 
    N = graphInitRDD.count()
    # Initialize mmAccum
    graphInitRDD.filter(lambda line: line[1][1]=='') \
                .foreach(lambda line: mmAccum.add(line[1][0])) 
    # The main while loop that keeps iteration going until either achieving
    # 'maxIter' or the probabilities of the next state changes less than 
    # the 'threshold' value. 
    while (maxIter > iteration):

        # Call inc_mmAccum to increment the missing mass and N accumulators.
        teleport_val = sc.broadcast(np.divide(a.value,N))
        m = mmAccum.value
        mmass_per_node = sc.broadcast(np.divide(m,N))
        
        # Use mappers to emit probability masses and use reducers to combine
        # the probability masses.
        steadyStateRDD = steadyStateRDD.flatMap(emit_prob_mass) \
                                       .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
                                       .flatMap(update_rank)\
                                       .cache()

        # Reset mmAccum to zero and then initialize it based on the current
        # iteration's values. 
        mmAccum = sc.accumulator(0.0, FloatAccumulatorParam()) 
        steadyStateRDD.foreach(inc_accum)
        # Compute the teleport portion of the rank and missing mass per node and
        # store these as broadcast variables for efficiency. Compute the rank  
        # and increment 'totAccum' by each rank to ensure the total rank is 1.

        if verbose:
            print('--->STEP {}: missing mass = {}, total = {}'.
                  format(iteration, m, totAccum.value))
       
        # Prepare for the next iteration by incrementing 'interation' 
        # and resetting totAccum.
        iteration += 1
        totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
  
    # Make a new RDD with the nodes and probability masses only.
    steadyStateRDD = steadyStateRDD.map(lambda line: (line[0], line[1][0]))
    
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

#testRDD = sc.textFile("gs://"+BUCKET+"/test_graph.txt")##
#nIter = 20
#testGraphRDD = initGraph(testRDD)
#start = time.time()
#test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True) ## verbose defaults to f
#print('...trained {} iterations in {} seconds.'.format(nIter,time.time() - start ))
#print('Top 20 ranked nodes:')
#test_results.takeOrdered(20, key=lambda x: - x[1])



nIter = 10
start = time.time()

# Initialize your graph structure (Q7)
wikiGraphRDD = initGraph(wikiRDD)
print('Finished Q7')
#print('Total number of records: {}'.format(wikiGraphRDD.count()))
#print('First record: {}'.format(wikiGraphRDD.take(1)))

# Run PageRank (Q8)
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)

print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
#print('...trained {} iterations in {} seconds.'.format(nIter,time.time() - start ))
#print('Top 20 ranked nodes:')
print(full_results.takeOrdered(20, key=lambda x: -x[1]))