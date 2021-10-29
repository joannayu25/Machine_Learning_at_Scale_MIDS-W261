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
'''
prev_word = ''
prev_count = 0
for line in sys.stdin:
    word, count = line.split()
    count = int(count)
    
    if prev_word =='': 
        prev_word = word
        prev_count = count
        print("if")
        
    elif prev_word == word: 
        prev_count += count
        print("elif")
    else:
        print("else")
        print("{}\t{}".format(prev_word, prev_count))
        prev_word = ''
        prev_count = 0
        
if len(sys.stdin.readlines())==1: print("{}\t{}".format(word, count))   
'''   

count = 0
lines = sys.stdin.readlines()

i = 0
while i < len(lines): 
    word, count = lines[i].split()
    count = int(count)
    i+=1
    
    if i < len(lines):
        print('current i', i)
        next_word, next_count = lines[i].split()
        next_count = int(next_count)
        print(word, next_word)

        if word==next_word:
            count += next_count
            i += 1
        else: 
            print("{}\t{}".format(word, count))
            break
            
else:
    print("{}\t{}".format(word, count))



################ (END) YOUR CODE #################
