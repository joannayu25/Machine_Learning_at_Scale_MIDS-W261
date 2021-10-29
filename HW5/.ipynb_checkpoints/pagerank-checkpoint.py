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
        yield (node, edges)
        for edge_node in edge_nodes.keys():
            yield (edge_node, "{}")

    def combineDict(x, y):
        if x=="{}": return y
        else: return x

    # write your main Spark code here
    
    # For all nodes in the data, emit (node_id , edges).
    # Use 'reduceByKey' to take out duplicates and cache 
    # this 'tempRDD'.
    tempRDD = dataRDD.flatMap(emit_nodes) \
                     .reduceByKey(combineDict) \
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
        # Parse the line, tokenize each element, and convert the
        # corresponding dictionary back to dict format. 
        print('in emit')
        node, (prob, edges) = line        
        if edges == '{}': yield node, (0, '{}') 
        else:
        
        # For each item in the neighboring node dictionary, calculate
        # the probability mass being redistributed and emit in the format
        # of (node_id, (score, {})) for each neighbor. To preserve the graph 
        # structure for the next iteration, we also emit the original node 
        # as (node_id, (0, {neighbors})).
            edge_dict = ast.literal_eval(edges)
            total_wt = sum(edge_dict.values())
            for edge_node in edge_dict.keys():
                current_wt = edge_dict[edge_node]
                yield (edge_node, (np.divide(prob,total_wt)*current_wt, '{}'))
            yield node, (0, str(edge_dict))
    
    def inc_mmAccum(line):
        node, (rank, edges) = line
        if edges == '{}': 
            mmAccum.add(rank)
        print('done inc mm')
    
    def combine_rank(a,b):

        a_rank, a_edges = a
        b_rank, b_edges = b
        combined_edges = ast.literal_eval(a_edges)
        combined_edges.update(ast.literal_eval(b_edges))
        print('combine rank', a, b, a_rank+b_rank)
        return a_rank+b_rank, str(combined_edges)
    

    def update_rank(line):
        node, (rank, edges) = line
        new_rank = teleport_val.value + d.value*(np.divide(m.value, N.value)+rank)
        yield node, (new_rank, edges)        
        
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

    # Initialize variables before the while loop. 'lastRDD' and 'currentRDD' are 
    # used to keep track of convergence. 'NAccum' is to keep track of N. 'Iteration'
    # is to keep track of the number of iterations so far. 'threshold' is the 
    # difference between the probabilities that defines convergence. 
    #lastRDD = None 
    steadyStateRDD = graphInitRDD
    iteration = 0
    N = sc.broadcast(graphInitRDD.count())
  #  threshold = sc.broadcast(1e-18)
    # The main while loop that keeps iteration going until either achieving
    # 'maxIter' or the probabilities of the next state changes less than 
    # the 'threshold' value. 
    while (maxIter > iteration):# and not_converged(lastRDD, currentRDD)):  
       # currentRDD = steadyStateRDD
        # Call inc_mmAccum to increment the missing mass and N accumulators.
      #  print(currentRDD.take(5))
        steadyStateRDD.foreach(inc_mmAccum)
      #  print('mmacum', mmAccum.value)

        teleport_val = sc.broadcast(np.divide(a.value,N.value))
        
        m = sc.broadcast(mmAccum.value)
        # Use mappers to emit probability masses and use reducers to combine
        # the probability masses.
        print('starting spark')
        steadyStateRDD = steadyStateRDD.flatMap(emit_prob_mass) \
                                       .reduceByKey(combine_rank) \
                                       .flatMap(update_rank)\
                                       .cache()
        print('finished spark and begin totaccum')
        
       # mmass_per_node = sc.broadcast(mmAccum.value)
       # print('#########',currentRDD.take(50))
      #  steadyStateRDD = tempRDD #.flatMap(update_rank) 
       # print(steadyStateRDD.take(11))
        #steadyStateRDD.foreach(lambda line: totAccum.add(line[1][0]))
        print('finish tot')
        
    #    print(steadyStateRDD.take(5))
        # Compute the teleport portion of the rank and missing mass per node and
        # store these as broadcast variables for efficiency. Compute the rank  
        # and increment 'totAccum' by each rank to ensure the total rank is 1.

        if verbose:
            print('--->STEP {}: missing mass = {}, total = {}'.
                  format(iteration, mmAccum.value, totAccum.value))
       
        # Prepare for the next iteration by incrementing 'interation' 
        # and resetting all accumulators.
        iteration += 1
        mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
        totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
  
    # Make a new RDD with the nodes and probability masses.
    steadyStateRDD = steadyStateRDD.map(lambda line: (line[0], line[1][0]))
    
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

nIter = 10
start = time.time()

# Initialize your graph structure (Q7)
wikiGraphRDD = initGraph(wikiRDD)
print('finished Q7')

# Run PageRank (Q8)
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)

print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
print(full_results.takeOrdered(20, key=lambda x: -x[1]))