#!/usr/bin/env python


import time
import numpy as np
import pyspark

from pyspark.sql import SQLContext
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.functions as f
# from pyspark.sql.functions import conv, mean, max, min
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, VectorIndexer, StringIndexer, StandardScaler, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import when
import re
import ast
import time
import numpy as np
import pandas as pd
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import lit,avg
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import SQLTransformer
import time
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import conv, mean, udf
from pyspark.sql.types import LongType, IntegerType, DoubleType, FloatType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, VectorIndexer, StringIndexer, StandardScaler, OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

app_name = "final_proj"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)
############## YOUR BUCKET HERE ###############

BUCKET="w261pw"

############## (END) YOUR BUCKET ###############


trainDF = spark.read.load("gs://"+BUCKET+"/toy_preprocessed.parquet")
validationDF = spark.read.load("gs://"+BUCKET+"/mini_preprocessed.parquet")

print(trainDF.count())
print(validationDF.count())

firstelement=udf(lambda v:float(v[0]),FloatType())
finalTrainDF = trainDF.select('Label',
                        firstelement('I-1-Scaled'), 
                             firstelement('I-2-Scaled')
                            , firstelement('I-3-Scaled')
                            , firstelement('I-4-Scaled')
                            , firstelement('I-5-Scaled')
                            , firstelement('I-6-Scaled')
                            , firstelement('I-7-Scaled')
                            , firstelement('I-8-Scaled')
                            , firstelement('I-9-Scaled')
                            , firstelement('I-10-Scaled')
                            , firstelement('I-11-Scaled')
                            , firstelement('I-13-Scaled')
                            , 'C-1-IndexB'
                            , 'C-2-IndexB'
                            , 'C-5-IndexB'
                            , 'C-6-IndexB'
                            , 'C-7-IndexB'
                            , 'C-8-IndexB'
                            , 'C-9-IndexB'
                            , 'C-10-IndexB'
                            , 'C-11-IndexB'
                            , 'C-13-IndexB'
                            , 'C-14-IndexB'
                            , 'C-15-IndexB'
                            , 'C-17-IndexB'
                            , 'C-18-IndexB'
                            , 'C-19-IndexB'
                            , 'C-20-IndexB'
                            , 'C-24-IndexB'
                            , 'C-25-IndexB'
                            , 'C-26-IndexB'
                            )

finalValidationDF = validationDF.select('Label',
                            firstelement('I-1-Scaled'), 
                             firstelement('I-2-Scaled')
                            , firstelement('I-3-Scaled')
                            , firstelement('I-4-Scaled')
                            , firstelement('I-5-Scaled')
                            , firstelement('I-6-Scaled')
                            , firstelement('I-7-Scaled')
                            , firstelement('I-8-Scaled')
                            , firstelement('I-9-Scaled')
                            , firstelement('I-10-Scaled')
                            , firstelement('I-11-Scaled')
                            , firstelement('I-13-Scaled')
                            , 'C-1-IndexB'
                            , 'C-2-IndexB'
                            , 'C-5-IndexB'
                            , 'C-6-IndexB'
                            , 'C-7-IndexB'
                            , 'C-8-IndexB'
                            , 'C-9-IndexB'
                            , 'C-10-IndexB'
                            , 'C-11-IndexB'
                            , 'C-13-IndexB'
                            , 'C-14-IndexB'
                            , 'C-15-IndexB'
                            , 'C-17-IndexB'
                            , 'C-18-IndexB'
                            , 'C-19-IndexB'
                            , 'C-20-IndexB'
                            , 'C-24-IndexB'
                            , 'C-25-IndexB'
                            , 'C-26-IndexB'
                            )

def vector_transform(DF, ignore_list):
    # Build vector for decision tree for dataset
    assembler = VectorAssembler(inputCols=[x for x in DF.columns if x not in ignore_list],
                                outputCol='features')

    # Transform the data for train data set
    output = assembler.transform(DF)
    #print(output.select("Label", "features").show(truncate=False))
    return output

# Compute log loss on the given Spark Dataframe.
def logLoss(predDF):
    # Define a function clamp to restrict the values of probability to be greater than 0 and less than one
    def clamp(n):
        epsilon = .000000000000001
        minn = 0 + epsilon
        maxn = 1 - epsilon
        return max(min(maxn, n), minn)
    
    # Define a UDF to extract the first element of the probability array returned which is probability of one
    firstelement=udf(lambda v:clamp(float(v[1])))   #,FloatType() after [] was inserted and removed for epsilon
    
    # Create a new dataframe that contains a probability of one column (true)
    predict_df = predDF.withColumn('prob_one', firstelement(predDF.probability))
    
    # Compute the log loss for the spark dataframe for each row
    row_logloss = (predict_df.withColumn(
        'logloss', -f.col('Label')*f.log(f.col('prob_one')) - (1.-f.col('Label'))*f.log(1.-f.col('prob_one'))))

    logloss = row_logloss.agg(f.mean('logloss').alias('ll')).collect()[0]['ll']
    return logloss

# Explore the optimal number of trees. 
RF_results = []

# Drop I-12 and C-22 with 76% missing values
#ignore = ['I-12-Scaled', 'C-22-IndexB', 'Label']
ignore = [ 'Label']

temp = ['Label']

finalTrainDF_trans = vector_transform(finalTrainDF, ignore)

finalValidationDF_trans = vector_transform(finalValidationDF, ignore)


print('DFs transformed for entry into RF Model')
print('Train')
print(finalTrainDF_trans.count())
print('Validation')
print(finalValidationDF_trans.count())

finalTrainDF_trans.persist(pyspark.StorageLevel.MEMORY_ONLY)
finalValidationDF_trans.persist(pyspark.StorageLevel.MEMORY_ONLY)


model = "Model #3, Random Forest - All Features"
data_size = "Full"
depth = 8
time0 = time.time()

# Train and test a Random Forest Classifier to make predicitions
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'Label', maxDepth = depth, numTrees=20)
print("model is fine")
rfModel = rf.fit(finalTrainDF_trans)
print("print is fine")
predictions = rfModel.transform(finalValidationDF_trans)
predictions.select('Label', 'rawPrediction', 'prediction', 'probability').show(10)

log_loss = logLoss(predictions)
evaluator = BinaryClassificationEvaluator(labelCol='Label')
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
wall_time = time.time()-time0
RF_results.append([depth, log_loss, auroc, auprc, wall_time])
print('Model = ', model, '  Data Size = ', data_size)
print('finished training')
RF_base_PD = pd.DataFrame(RF_results, columns=['Depth', 'Log Loss', 'Area Under ROC', 'Area Under PR', 'Wall Time'])
print(RF_base_PD)
print()
print(RF_results)

print(predictions.show(10))

# Metrics



#X needs to be which column the prediction is in - 43 in full, 17 in integer only
x = predictions.rdd.map(lambda x:(x[17], float(x[0]))).collect()
predictionAndLabels = sc.parallelize(x)

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
fpr0 = metrics.falsePositiveRate(0.0)
fpr1 = metrics.falsePositiveRate(1.0)
accuracy = metrics.accuracy
print("Summary Statistics")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 score = %s" % f1Score)
print("False positive rate 0 = %s" % fpr0)
print("False positive rate 1 = %s" % fpr1)
print("Accuracy = %s" % accuracy, '\n')
print("Confusion Matrix")
print(metrics.confusionMatrix().toArray(), '\n')
