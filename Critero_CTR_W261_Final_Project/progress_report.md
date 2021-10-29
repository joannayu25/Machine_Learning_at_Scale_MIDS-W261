## Final Project Progress Report (W261 Fall 2019, Team 19)

### Accomplishments
Our group has been meeting twice a week to discuss progress and challenges on each member's tasks. We have a Slack channel to discuss project issues and post relevant research articles. 


Pauline:
* Loaded data and setup Google share drive for collaboration. 
* Built a toy data set using a subset of the training.txt.
* Built a decision tree classifier and random forest using the toy data set. 
* Experimented with data preprocessing - bin numeric features and using the Breiman theorem for the categorical features. 
    
Steve:
* Drafted our goals for the project in part 1 for CTA.  
* Downloaded all data and worked on data conversions.  
* Set up Jupyter Notebook in Docker for SparkML experiments.  
* Coded transformations of data to make numeric data to fit in VectorAssembler and eliminate columns.  
* Built Spark ML model and successfully executed toy data decision tree.  
* Working on scaling up the data sizes in Spark ML now.  

Naga:
* Researched Parquet conversion and created Parquet files in GCP for training dataset (80%), validation dataset (20%), and toy dataset (2% of training dataset).
* Created various code snippets to load the data from Parquet format and cast them to usable formats. 
* Researched various aspects of the project and provided findings to the group.
* Helped troubleshoot other team members' issues. 

Joanna:
* Performed preliminary EDA on the 80% training dataset using RDD.
* Experimented with Parquet conversion using RDD and Spark Dataframe. 
* Once the Parquet files became available, performed additional EDA and visualizations using Spark Dataframe and Pandas.
* Began the write-up for the EDA section of the final report.

### Plans
The group will continue to meet twice a week and Slack regularly. Everyone is expected to help with the final report write-up and presentation slides. 

Pauline:
* Discuss and finalize a data preprocessing plan with the team so we can implement it on the entire train.txt file. 
* Write out the algorithm implementation section include rational of selecting DT, entropy calculation, data preprocess formula. 
* Figure log-loss calculation and how we can benchmark against other Kaggle teams in the same competition.
* Assist build/implement the algorithm for the entire dataset. 
* Finalize the write up with the group for section I (question formulation) and question V (course lesson application)

Steve:
* Convert the model to random forest and scale data size, develop metrics and code for measuring the success of the decision tree/random forest and write edit as needed in the project.

Naga:
* Wrap up EDA with Joanna. 
* Explored how different data pre-processing may improve the model. 

Joanna:
* Wrap up the EDA section and finish the EDA write-up.
* Explored how different data pre-processing may improve the model. 
