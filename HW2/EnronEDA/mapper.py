#!/usr/bin/env python
"""
Mapper tokenizes and emits words with their class.
INPUT:
    ID \t SPAM \t SUBJECT \t CONTENT \n
OUTPUT:
    word \t class \t count 
"""
import re
import sys

# read from standard input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    
############ YOUR CODE HERE #########
    # For each word in the document, emit the word, its class, 
    # and a count of 1 with tab delimiter. 
    for word in words:
        print('{}\t{}\t{}'.format(word, _class, 1))

############ (END) YOUR CODE #########