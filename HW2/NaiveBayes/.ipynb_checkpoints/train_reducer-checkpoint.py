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

# Initialize trackers
current_word = None
ham_totalCount, spam_totalCount, ham_prob, spam_prob = 0,0,0,0
ham_docs, spam_docs, ham_words, spam_words = 0,0,0,0

# Read from standard input
for line in sys.stdin:
    # Parse input
    pKey, word, partialCounts = line.split('\t')
    #print(pKey, word, partialCounts)
    # Convert strings of counts to floats
    ham_partialCount = float(partialCounts.split(',')[0])
    spam_partialCount = float(partialCounts.split(',')[1])
    # Define a lambda function to take the log
    take_log = lambda x: np.log(x) if x != 0 else float("-inf")

    # If *docTotals is encountered, calculate the class priors.
    if word == '*docTotals': 
        ham_docs += ham_partialCount
        spam_docs += spam_partialCount
       # ham_classPrior = ham_docs / (ham_docs + spam_docs)
       # spam_classPrior = spam_docs / (ham_docs + spam_docs)
    
    # If *wordTotals is encountered, save the total number of ham 
    # words and spam words into two variables. 
    elif word == '*wordTotals':
        ham_words += ham_partialCount
        spam_words += spam_partialCount
    
    # If a new word has not been encountered, update the class total 
    # counts (spam_totalCount and ham_totalCount).
    elif current_word == None or current_word == word:
        current_word = word
        ham_totalCount += ham_partialCount
        spam_totalCount += spam_partialCount
        
    # If a new word has been encountered, update the class total counts
    # and calculate the log probabilities and print the word along with 
    # the 4 values. 
    else: 
        ham_prob = ham_totalCount / ham_words
        spam_prob = spam_totalCount / spam_words
        print('{}\t{}\t{}\t{}\t{}'
              .format(current_word, ham_totalCount, spam_totalCount,
                     ham_prob, spam_prob))
        ham_totalCount, spam_totalCount, ham_prob, spam_prob = 0,0,0,0
        ham_totalCount += ham_partialCount
        spam_totalCount += spam_partialCount
        current_word = word
        
ham_prob = ham_totalCount / ham_words
spam_prob = spam_totalCount / spam_words
print('{}\t{}\t{}\t{}\t{}'.format(current_word, ham_totalCount, spam_totalCount,
                                  ham_prob, spam_prob))
        
        
        


































##################### (END) CODE HERE ####################