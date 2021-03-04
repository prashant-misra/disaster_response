# Disaster Response Pipeline Project
I have used basic Pipeline skills to process and distinguish texts received from a disaster spot. Here I have used data provided by Figure Eight and build an model for an App that classifies text messages during need of hour to proper place.

### Table of Contents:
1) Description
2) Instructions
3) File Description
4) Acknowledgement


### Description
This project is used based on text data provided by Figure Eight. It has an web app that receives text message, process it and classify it to the group according to the need.
It has three steps:
    a. Taking text from web app, process it and through an ETL pipeline and then storing the data in a SQLite Database.
    b. Loading data from database, during natural language processing and building a ML Model that classfy the message to proper category.
    c. Finally,there is a web app using Flask to take the user input.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### File Description:
This project contain main 3 python files.

process_data.py: It load data data from disaster_messages.csv and disaster_categories.csv file, merges it, cleabn it and load cleaned data into database.

train_classifier.py: It load the data from database and tokenize,lemmatize the text data. It also build a ML Model(Here RandomClassifier) to categorize the data according to categories available.

run.py: It contain code for the web app. It recieves the text data from user to get it classified according to need.

This Project contains two csv filrs:

Disaster_messages.csv: It contain the Disater messages provided by Figure Eight.

Disaster_Categories.csv: It contain data abbot the various categories between which text need to be classified.


### Acknowledgement:
Udacity for providing scope for this project.
Figure Eight for providing the data used in this project
