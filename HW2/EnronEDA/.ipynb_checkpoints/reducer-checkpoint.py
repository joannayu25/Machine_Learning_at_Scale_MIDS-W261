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
    
    if current_word == None or current_word == word: 
        current_word = word
        if is_spam: spam_count+=count
        else: ham_count+=count
       
    else: 
        print('{}\t{}\t{}'.format(current_word,1,spam_count)) 
        print('{}\t{}\t{}'.format(current_word,0,ham_count))      
        current_word = word
        spam_count, ham_count = 0,0
        if is_spam: spam_count=count
        else: ham_count=count      
            
print('{}\t{}\t{}'.format(current_word,1,spam_count))  
print('{}\t{}\t{}'.format(current_word,0,ham_count)) 

'''
    count = int(count)
    is_spam = int(is_spam)
    
    if current_word == None: 
        current_word = word
        if is_spam: spam_count=count
        else: ham_count=count
            
    elif word == current_word:
        if is_spam: spam_count += count
        else: ham_count += count
       
    else: 
        print('{}\t{}\t{}'.format(current_word,1,spam_count)) 
        print('{}\t{}\t{}'.format(current_word,0,ham_count))      
        current_word = word
        spam_count, ham_count = 0,0
        if is_spam: spam_count=count
        else: ham_count=count      
            
print('{}\t{}\t{}'.format(current_word,1,spam_count))  
print('{}\t{}\t{}'.format(current_word,0,ham_count)) 
'''

'''
    count = int(count)
    is_spam = int(is_spam)
    
    if current_word == None: 
        current_word = word
        if is_spam: spam_count=count
        else: ham_count=count
            
    elif word == current_word:
        if spam_count & is_spam: spam_count += count
        elif ham_count & (not is_spam): ham_count += count
        elif spam_count: 
            print('{}\t{}\t{}'.format(current_word,1,spam_count))
            spam_count = 0
            ham_count = count
        elif ham_count:
            print('{}\t{}\t{}'.format(current_word,0,ham_count))
            spam_count = count
            ham_count = 0         
    else: 
        if spam_count:
            print('{}\t{}\t{}'.format(current_word,1,spam_count))  
        else: 
            print('{}\t{}\t{}'.format(current_word,0,ham_count))
        current_word = word
        if is_spam: spam_count=count
        else: ham_count=count
            
print('{}\t{}\t{}'.format(current_word,is_spam,spam_count+ham_count))  
'''

############ (END) YOUR CODE #########