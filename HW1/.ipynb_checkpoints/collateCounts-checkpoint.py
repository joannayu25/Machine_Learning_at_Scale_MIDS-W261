#!/usr/bin/env python
"""
This script reads word counts from STDIN and combines
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE:
    python collateCounts.py < yourCountsFile.txt

Instructions:
    For Q6 - Use the provided code as is. (you'll need to uncomment it)
    For Q7 - Delete or comment out the section marked "PROVIDED CODE" &
             replace it with your own implementation. Your solution 
             should not use a dictionary or store anythin other than a 
             signle total count - just print them as soon as you've 
             added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys
from collections import defaultdict

########### PROVIDED IMPLEMENTATION ##############  
##### uncomment to run
'''
counts = defaultdict(int)
# stream over lines from Standard Input
for line in sys.stdin:
    # extract words & counts
    #print(line.split())
    #print(len(line.split()))
    word, count  = line.split()
     # tally counts
    counts[word] += int(count)
# print counts
for wrd, count in counts.items():
    print("{}\t{}".format(wrd,count))
'''
########## (END) PROVIDED IMPLEMENTATION #########

################# YOUR CODE HERE #################

prev_word = ''
prev_count = 0
lines = sys.stdin.readlines()
# Deals with countfiles with 1 line, where no comparison
# is needed
if len(lines) == 1: 
    word, count = line[0].split()
    print("{}\t{}".format(word, int(count)))

for line in lines:
    word, count = line.split()
    count = int(count)
    
    if prev_word =='': 
        prev_word = word
        prev_count = count
        
    elif prev_word == word: prev_count += count
        
    else:
        print("{}\t{}".format(prev_word, prev_count))
        prev_word = word
        prev_count = count


################ (END) YOUR CODE #################
