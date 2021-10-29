#!/usr/bin/env python
"""
Reducer aggregates word counts by class and emits frequencies.
    
INPUT:
    partitionKey \t word \t class0_partialCount,class1_partialCount
OUTPUT:
    word \t ham_totalCount \t spam_totalCount \t P(ham|doc) \t P(spam|doc)
    
    
Instructions:
    Again, you are free to design a solution however you see 
    fit as long as your final model meets our required format
    for the inference job we designed in Question 8. Please
    comment your code clearly and concisely.
    
    A few reminders: 
    1) Don't forget to emit Class Priors (with the right key).
    2) In python2: 3/4 = 0 and 3/float(4) = 0.75
"""
##################### YOUR CODE HERE ####################

import re
import sys
import numpy as np

# helper function to emit records correctly formatted
def EMIT(*args):
    print('{}\t{},{},{},{}'.format(*args))
    
# Initialize trackers
current_word = None
current_counts = np.array([0,0])
docTotals = np.array([0.0,0.0])
wordTotals = np.array([0.0,0.0])

# Read from standard input
for line in sys.stdin:
    # Parse input
    pKey, word, partialCounts = line.split('\t')
    partialCounts = [int(c) for c in partialCounts.split(',')]
    
    # Define a lambda function to take the log
    take_log = lambda x: np.log(x) if x != 0 else float("-inf")

    # If *docTotals is encountered, calculate the class priors.
    if word == '*docTotals': 
        docTotals += partialCounts
    
    # If *wordTotals is encountered, save the total number of ham 
    # words and spam words into two variables. 
    elif word == '*wordTotals':
        wordTotals += partialCounts
    
    # If a new word has not been encountered, update the class total 
    # counts (spam_totalCount and ham_totalCount).
    elif current_word == None or current_word == word:
        current_word = word
        current_counts += partialCounts
        
    # If a new word has been encountered, update the class total counts
    # and calculate the log probabilities and print the word along with 
    # the 4 values. 
    else: 
        prob = current_counts / wordTotals
        EMIT(current_word, *tuple(current_counts)+tuple(prob))
        current_counts = np.array(partialCounts)
        current_word = word

if current_word != None:
    prob = current_counts / wordTotals
    EMIT(current_word, *tuple(current_counts)+tuple(prob))
    
    priors = tuple(docTotals) + tuple(docTotals/sum(docTotals))
    EMIT('ClassPriors', *priors)
        
        
        


































##################### (END) CODE HERE ####################