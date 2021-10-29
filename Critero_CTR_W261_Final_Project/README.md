# CriteoLab Click-Through-Rate

Course: w261 Final Project (Criteo Click Through Rate Challenge)

Team Number:19

Team Members: Steve Dille, Naga Akkineni, Joanna Yu, Pauline Wang

Fall 2019: section 1, 3, 4

## Introduction
Our goal in this machine learning project is to **build a model that can predict advertising click through rates (CTR)** better than a baseline model of guessing based on known click through percentages in the data set or by using an out-of-the box machine learning model. We seek to add value to the process. This is an enormously valuable task which is why there is so much investment in machine learning and data science in the advertising area. Being able to have better predictions of what the factors that drive clickthrough rates are is valuable to both the buy side (advertising buyers) and the sell side (advertisement platforms or publishers). If advertisers have better predictive power of clickthrough conversion rates, they can use this to determine expected sales from a campaign, where to advertise, how much they can spend as a profitable investment, or which platforms and publishers to advertise with.  On the sell side, if we are able to model and predict conversions, we could adjust pricing models to suit the advertisers who are interested in specific advertisement products and achieve higher revenue and greater customer satisfaction.

## About the Data
The provided training file contains **~46 million data points (45,840,617 rows in a 11GB file)**. There are 39 unlabeled features containing 1) 13 integer features (I-1 to I-13) with values range from -2 to 1.3 millions and distributions with various degree of skewness and kurotisis. 2) 26 categorical features that are hashed for anonymization. 

We use Brieman's theorem to process the unordered categorical feature in order to find the best split predicate for a categorical attribute, without evaluation all possible subsets.  In order to facilitate the process of calculating Brieman, we first rank the categorical values by their frequency, and only use the top 100 most frequently occur values. We label any categorical feature values that are not on the top 100 table as 999. For more information about how StringIndexer, please reference SparkMLlib Documentation [1] 

## Algorithm Explanation

We choose **decision tree and random forest** as our algorithm for the Criteo project because of the nature of the dataset we are given.  First of all, the data set contains both continuous numeric features and categorical features.  Second, our Exploratory Data Analysis reveals that the data set contains many missing values. Decision tree/random forest algorithms are known to be robust in handling of data sets with mixed features and missing values.  Last but not least, we are not given any information about what the features represent, and therefore, we want to keep the data preprocessing and feature selection to a minimum by simply grouping feature values in representative bins.

## Metric - Log loss

## Conclusion

This project is our first hands-on experience solving a machine learning challenge on a large dataset. We were able to apply many of the techniques we learned from class to guide our data exploration and modeling choice. Throughout our design process, we thought we had made scalability a priority and used distributed systems like Spark DataFrame and Spark ML operations on Parquet data. Everything ran properly for the toy dataset. However, as we ran the big dataset in GCP, it became clear that every small inefficiency has big implications for big data. We had researched every step we took in the pre-processing stage and changed our design so data transformation and selection can be more scalable. Some of the issues we found include:
1. String Indexer turned out to be inefficient for some cases since it has to rank the unique values based on frequency. For some of the features with millions of unique values, this step took excessively amount of time so we ignored those features. 
2. When we have many partitions, broadcast join (mapper side join) is highly inefficient. We had previously used that to implement Breiman but had changed to SQLTransformer. 
3. We discovered that simple operations like 'withColumn' on the DataFrame actually does not perform filtering as we intended. It actually creates a brand new DataFrame so we used SQLTransformer instead. 
4. For Breiman, we partition over a column and apply average. We found that using Window functions allows us to more efficiently calculate a moving average over a range of input rows. 
5. To avoid overwhelming the computing resources, we run pre-processing independently and store them as Parquet so the pre-processed data can be fed to the model directly. 

This dataset is the largest dataset we had encountered in this class. Although it is not anywhere near the petabyte scale, we had our first taste of what processing big data really means. The entire data processing and modeling pipeline will need to be carefully designed to handle the scalability challenges. 

We are pleased that we were able to get all of our code and models to run and prove that our modeling strategy and algorithms work. In the end, we had to settle for running model IV and V on a subset of data. With more time we would have experimented more with how to pre-process the data to better bin data and optimize the number of splits the decison tree algorithm would have to run. We would have also liked to optimized our data set by reducing the features that would have enabled better scalability. We determined that our point of failure in scaling was in model fitting.  We could have had better scalability with larger clusters and more memory. 

## Our final submission comprised of the following components:
* Jupyter notebook - main Jupyter notebook of our project.
* .py files - Python files used for GCP submission to run the models in the cloud.
> 1. **'steve_ctr_DT_full.py'** - GCP job for Decision Tree base model (Model I) with only integer features on the **FULL** training set.
> 2. **'final_proj_GCP_RF_base.py'** - GCP job for Random Forest base model (Model II) with only integer features on the **TOY** dataset.
> 3. **'steve_ctr_RF_full.py'** - GCP job for Random Forest base model (Model II) with only integer features on the **FULL** training set.
> 4. **'steve_ctr_GBT_full.py'** - GCP job for Random Forest model with only integer features plus Gradiet Boost (Model III) on the **FULL** training set.
> 4. **'final_proj_GCP_RF_SSI2.py'** - GCP job for Random Forest model with pre-processing, meaning scaled integer features and string-indexed, threshold controlled categorical features (Model IV) on the **TOY** dataset.
> 5. **'rf.py'** - GCP job for Random Forest model with pre-processing, meaning scaled integer features and string-indexed, threshold controlled categorical features (Model IV) on the **FULL** training set.
> 6. **'gcb.py'** - GCP job for Final Random Forest model with preprocessing listed in Model IV plus Gradient Boosting and Breiman (Model V) on the **FULL** training set.

* Pickle files - Pickle files for the Pandas tables that we created for pretty printing:
> 1. **'summary.pkl'** stores the summary statistics table for the training dataset.
> 2. **'correlation.pkl'** stores all the pairwise correlation values for all 39 features and the label for the training dataset.
> 3. **'correlation_subset.pkl'** takes the table from 'correlation.pkl' and filter out entries with correlation values > 0.5.
> 4. **'DT_base_PD.pkl'** stores the summary metrics table for the Decision Tree base model with varied MaxDepth to compare model performance.