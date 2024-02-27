# -*- coding: utf-8 -*-
"""NBA_Multi_Category_Logistic_Regression_Model

created by Brian Ramm

Original colab notebook file is located at
    https://colab.research.google.com/drive/14vsQ7h5_N26Q9jGsyPAnscVH26sibSOR
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import *

from pyspark.ml.feature import CountVectorizer, VectorAssembler
from pyspark.ml import Pipeline



def evaluate_predictions(input_data, label_col, prediction_col):

    true_positive = input_data.filter((F.col(label_col) == 1) & (F.col(prediction_col) == 1)).count()
    false_positive = input_data.filter((F.col(label_col) == 0) & (F.col(prediction_col) == 1)).count()

    true_negative = input_data.filter((F.col(label_col) == 0) & (F.col(prediction_col) == 0)).count()
    false_negative = input_data.filter((F.col(label_col) == 1) & (F.col(prediction_col) == 0)).count()

    # Print the Contingency matrix
    print("--Contingency matrix--")
    print(f" TP:{true_positive:6}  FP:{false_positive:6}")
    print(f" FN:{false_negative:6}  TN:{true_negative:6}")
    print("----------------------")

    # Calculate the Accuracy and the F1
    accuracy = (true_positive + true_negative) / input_data.count()
    f1 = true_positive / (true_positive + 0.5 * (false_positive + false_negative))
    print(f"Accuracy = {accuracy}  \nF1 = {f1}")

#Main
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)

    """### Lets try spark dataframe"""
    sc = SparkContext(appName="Final-Project")
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # import data
    train_spark_df = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(sys.argv[1])
    test_spark_df = sqlContext.read.format('csv').options(header='true', inferSchema='true',  sep =",").load(sys.argv[2])


    columns = ['game_id', 'team_id', 'a_team_id', 'h_price', 'a_price', 'is_home', 'wl',
        'season_year', 'season_num', 'player_1', 'player_2', 'player_3', 'player_4',
        'player_5', 'player_6', 'team_id_a', 'player_1_a', 'player_2_a',
        'player_3_a', 'player_4_a', 'player_5_a', 'player_6_a']

    # Create a new column with the desired format
    processed_train_df = train_spark_df.withColumn("h_players", array([col(word) for word in columns[9:15]]))
    processed_train_df = processed_train_df.withColumn("a_players", array([col(word) for word in columns[17:]]))

    # Select only the relevant columns
    processed_train_df = processed_train_df.select('game_id', 'team_id', 'a_team_id', 'h_price', 'a_price', 'is_home', 'wl',
        'season_year', 'season_num', 'h_players', 'team_id_a', 'a_players')

    # Create a new column with the desired format
    processed_test_df = test_spark_df.withColumn("h_players", array([col(word) for word in columns[9:15]]))
    processed_test_df = processed_test_df.withColumn("a_players", array([col(word) for word in columns[17:]]))

    # Select only the relevant columns
    processed_test_df = processed_test_df.select('game_id', 'team_id', 'a_team_id', 'h_price', 'a_price', 'is_home', 'wl',
        'season_year', 'season_num', 'h_players', 'team_id_a', 'a_players')


    """### Create SVM Pipeline"""

    player_pool_size = 300

    # Create a count vectorizer
    countVectorizer = CountVectorizer(inputCol="h_players", outputCol="Features_h", vocabSize=player_pool_size)
    countVectorizer_away = CountVectorizer(inputCol="a_players", outputCol="Features_a", vocabSize=player_pool_size)

    from pyspark.ml.classification import LinearSVC

    # train models
    lsvc = LinearSVC(featuresCol=countVectorizer.getOutputCol(), labelCol='wl', predictionCol='HomePred', rawPredictionCol='rawHomePred', maxIter=20)

    away_lsvc = LinearSVC(featuresCol=countVectorizer_away.getOutputCol(), labelCol='wl', predictionCol='AwayPred', rawPredictionCol='rawAwayPred', maxIter=20)

    # Create a preprocessing pipeline with 4 stages
    pipeline_p = Pipeline(stages=[countVectorizer, countVectorizer_away, lsvc, away_lsvc])

    # Learn the data preprocessing model
    data_model = pipeline_p.fit(processed_train_df)

    # Transform Train Data
    transformed_data = data_model.transform(processed_train_df)
    transformed_data.toPandas().head(5)

    # Transform Test Data
    transformed_test_data = data_model.transform(processed_test_df)
    transformed_test_data.toPandas().head(5)

    # Get the vocabulary of the CountVectorizer
    # These are the most referenced basketball players
    print("Most common players:")
    print(data_model.stages[0].vocabulary[:10])

    """### Train SVM Model on player data

    ### Evaluate Our Model on home/away players seperate
    """
    print("Player based SVM:")
    evaluate_predictions(transformed_test_data, "wl", prediction_col="HomePred")

    evaluate_predictions(transformed_test_data, "wl", prediction_col="AwayPred")

    """Looks okay both around 61%"""

    """### What if we apply a logistic regression model on the raw values"""

    # Create a count vectorizer for each set of players
    countVectorizer = CountVectorizer(inputCol="h_players", outputCol="Features_h", vocabSize=player_pool_size)
    countVectorizer_away = CountVectorizer(inputCol="a_players", outputCol="Features_a", vocabSize=player_pool_size)

    # combine player vectors and season
    assembler = VectorAssembler(inputCols=["Features_h", "Features_a", "season_num"], outputCol="featureVector")

    # train models
    lsvc = LinearSVC(featuresCol=assembler.getOutputCol(), labelCol='wl', predictionCol='SVMPred', rawPredictionCol='rawSVMPred', maxIter=20)

    # Create a preprocessing pipeline with 4 stages
    multi_feature_p = Pipeline(stages=[countVectorizer, countVectorizer_away, assembler, lsvc])

    # Learn the data preprocessing model
    multi_feature_model = multi_feature_p.fit(processed_train_df)

    # Transform Train Data
    output_train_data = multi_feature_model.transform(processed_train_df)

    # Transform Test Data
    output_test_data = multi_feature_model.transform(processed_test_df)

    """# Multi Column SVM Results"""
    
    print("Multi Column SVM:")
    evaluate_predictions(output_test_data, "wl", prediction_col="SVMPred")

    """# Okay Now try Logistic Regression"""

    from pyspark.ml.classification import LogisticRegression

    logR = LogisticRegression(featuresCol=assembler.getOutputCol(), labelCol='wl', predictionCol='LogPred', rawPredictionCol='rawLogPred', maxIter=20)

    # Create a preprocessing pipeline with 4 stages
    multi_feature_logR = Pipeline(stages=[countVectorizer, countVectorizer_away, assembler, logR])

    # Learn the data preprocessing model
    multi_feature_logR_model = multi_feature_logR.fit(processed_train_df)

    # Transform Train Data
    output_train_data = multi_feature_logR_model.transform(processed_train_df)

    # Transform Test Data
    output_test_data = multi_feature_logR_model.transform(processed_test_df)

    """# Multi Column Logistic Reg Results"""

    print("Multi Column Logistic Regression:")
    evaluate_predictions(output_test_data, "wl", prediction_col="LogPred")

    """Compare logistic Regression model raw outputs with betting lines"""

    # Find the profit with the given prediction model just on wins

    def get_profit_basic(row):
        """Accumulate profit for every game with correct prediction"""
        profit = 0
        if row["wl"] == row["LogPred"]:
            if row["wl"] == 0:
                profit = row["a_price"] - 1
            elif row["wl"] == 1:
                profit = row["h_price"] - 1
        else:
            profit = -1
        return profit

    # Define a UDF (User-Defined Function)
    get_profit_basic_udf = F.udf(get_profit_basic)

    # apply on df
    output_test_data = output_test_data.withColumn("profit", get_profit_basic_udf(F.struct("wl", "LogPred", "a_price", "h_price")))

    sum_result = output_test_data.select(sum("profit")).collect()[0][0]
    # check results
    print(f"Simple game betting results in a profit of: {sum_result}")

    yearly_result = output_test_data.groupBy("season_year").agg(sum("profit").alias("yearly_sum"))

    # save results
    yearly_result.coalesce(1).write.csv(sys.argv[3])

    """results with probability based decision"""

    # helper row functions
    firstelement=udf(lambda v:float(v[1]),FloatType())
    home_odds=udf(lambda x: 1/x)
    away_odds=udf(lambda x: 1/x)

    # get model probability and convert gambling lines to probability 
    output_test_data = output_test_data.withColumn("h_win_probability", firstelement('probability'))
    output_test_data = output_test_data.withColumn("h_book_prob", home_odds("h_price"))
    output_test_data = output_test_data.withColumn("a_book_prob", away_odds("a_price"))

    # Find the profit with the given prediction model just on wins
    def get_profit(row):
        """More complicated betting function that places bet based on probability vs odds"""
        profit = 0

        if float(row["h_win_probability"]) >= float(row["h_book_prob"]):
            if row["wl"] == 1:
                profit = row["h_price"] - 1
            else:
                profit = -1

        elif float(row["h_win_probability"]) < float(row["a_book_prob"]):
            if row["wl"] == 0:
                profit = row["a_price"] - 1
            else:
                profit = -1

        return profit
    
    # Define a UDF (User-Defined Function)
    get_profit_udf = F.udf(get_profit)

    # apply on df
    output_test_data = output_test_data.withColumn("odds_profit", get_profit_udf(F.struct("wl", "h_win_probability", "h_book_prob", "h_price", "a_book_prob", "a_price")))

    sum_result = output_test_data.select(sum("odds_profit")).collect()[0][0]
    # check results
    print(f"Simple game betting results in a profit of: {sum_result}")

    yearly_result = output_test_data.groupBy("season_year").agg(sum("odds_profit").alias("yearly_sum"))

    # save results
    yearly_result.coalesce(1).write.csv(sys.argv[4])

    sc.stop()