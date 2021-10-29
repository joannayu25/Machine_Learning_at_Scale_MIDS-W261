#!/usr/bin/env python
"""
Reducer to calculate precision and recall as part
of the inference phase of Naive Bayes.
INPUT:
    ID \t true_class \t P(ham|doc) \t P(spam|doc) \t predicted_class
OUTPUT:
    precision \t ##
    recall \t ##
    accuracy \t ##
    F-score \t ##
         
Instructions:
    Complete the missing code to compute these^ four
    evaluation measures for our classification task.
    
    Note: if you have no True Positives you will not 
    be able to compute the F1 score (and maybe not 
    precision/recall). Your code should handle this 
    case appropriately feel free to interpret the 
    "output format" above as a rough suggestion. It
    may be helpful to also print the counts for true
    positives, false positives, etc.
"""
import sys

# initialize counters
FP = 0.0 # false positives
FN = 0.0 # false negatives
TP = 0.0 # true positives
TN = 0.0 # true negatives

# read from STDIN
for line in sys.stdin:
    # parse input
    docID, class_, pHam, pSpam, pred = line.split()
    # emit classification results first
    print(line[:-2], class_ == pred)
    
    # then compute evaluation stats
#################### YOUR CODE HERE ###################
    class_ = int(class_)
    pred = int(pred)
    if class_ == pred == 1: TP+=1
    elif class_ == pred == 0: TN+=1
    elif class_ == 1 and pred == 0: FN+=1
    else: FP+=1

accuracy = (TP+TN)/(FP+FN+TP+TN)
precision = TP/(TP+FP)
recall = TP/(FN+TP)
fscore = 2*(precision*recall)/(precision+recall)
print('# Documents:\t{:.0f}\n'\
      'True Positives:\t{:.0f}\n'\
      'True Negatives:\t{:.0f}\n'\
      'False Positives:\t{:.0f}\n'\
      'False Negatives:\t{:.0f}\n'\
      'Accuracy\t{:.4f}\n'\
      'Precision\t{:.4f}\n'\
      'Recall\t{:.4f}\n'\
      'F-Score\t{:.4f}'
      .format(FP+FN+TP+TN,TP,TN,FP,FN,accuracy,precision,recall,fscore))
     






















#################### (END) YOUR CODE ###################
    