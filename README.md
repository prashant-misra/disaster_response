# Disaster Response Pipeline Project -- Prashant Misra
Fixed and reuploaded 3rd march 2021

# Added as a fix based on reviewer comments

After a disaster, there are a number of different problems that may arise. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, so predictive modeling can help classify different messages more efficiently.Using a dataset containing 30,000 categorized messages from real natural and humanitarian disasters, the project uses natural language processing and machine learning to classify messages into categories. 

I have used basic Pipeline skills to process and distinguish texts received from a disaster spot. Here I have used data provided by Figure Eight and build an model for an App that classifies text messages during need of hour to proper place.

# License (Added to improve readibilty)
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.


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

This Project contains two csv files:

Disaster_messages.csv: It contain the Disater messages provided by Figure Eight.

Disaster_Categories.csv: It contain data abbot the various categories between which text need to be classified.


### Acknowledgement:
Udacity for providing scope for this project.
Figure Eight for providing the data used in this project
