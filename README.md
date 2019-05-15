# DSND-Disaster-Response
## Project Overview
This Project is part of Data Science Nanodegree Program by Udacity.
It uses the data set in which disaster messages are received by aid organizations.
A multi-output DecisionTreeClassifier is trained by supervised learning of natural language processing (NLP).

I created an ETL pipeline to extract data from the CSV-file, clean it up and load into the sql-database.
Then a machine learning pipeline is established to extract NLP features and optimize the algorithm by using grid-search. 
A Web application is developed to extract initial data from the database and provide some interactive visual summaries.
Users can also input their own messages, to classify by algorithm and get important keywords.

## Files
- app 
  - run.py
  - templates
    - go.html
    - master.html

- data
  - DisasterResponse.db
  - disaster_categories.csv
  - disaster_messages.csv
  - process_data.py

- model
  - classifier.pkl
  - train_classifier.py
  
- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb
- README.md

## Main libraries
- numpy
- pandas
- plotly
- sikitlearn
- pickle
- nltk
- Flask
- gunicorn
- sqlalchemy

## Instructions:
```
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
```
