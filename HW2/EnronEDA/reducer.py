#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########
    # Convert string to integers for numerical fields
    count = int(count)
    is_spam = int(is_spam)
    
    # If a new word has not been encountered, increment
    # spam_count and ham_count. 
    if current_word == None or current_word == word: 
        current_word = word
        if is_spam: spam_count+=count
        else: ham_count+=count
    
    # If a new word has been encountered, print spam_count
    # and ham_count, reset the two counts, and update the 
    # current word. 
    else: 
        print('{}\t{}\t{}'.format(current_word,1,spam_count)) 
        print('{}\t{}\t{}'.format(current_word,0,ham_count))      
        current_word = word
        spam_count, ham_count = 0,0
        if is_spam: spam_count=count
        else: ham_count=count      
            
# Print the very last word and its counts. 
print('{}\t{}\t{}'.format(current_word,1,spam_count))  
print('{}\t{}\t{}'.format(current_word,0,ham_count)) 

############ (END) YOUR CODE #########