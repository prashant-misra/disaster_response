# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load categiries and messages data from csv file as dataframe
    Input:
        messages_filepath: File path of disaster_messages.csv
        categories_filepath: File path of disaster_categories.csv
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
    # Loading message csv file
    messages = pd.read_csv(messages_filepath)
    
    # Loding categories csv file
    categories = pd.read_csv(categories_filepath)
    
    # Creating dataframe merging messages and categories dataframes
    df = pd.merge(messages,categories,how='outer',on=['id'])
    
    return df


def clean_data(df):
    '''
    Cleaning the dataframe 
    Input:
        df: Dataframe to be cleaned
    Output:
        df: Cleaned Dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0][:]
    
    #getting column names 
    category_colnames = row.str.split('-')
    category_colnames = list(category_colnames.apply(lambda x: x[0]).values)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Changing str to num
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # Checking and removing duplicates
    duplicates = df[df.duplicated() == True]
    
    # drop duplicates
    df.drop(list(duplicates.index),axis = 0,inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Save the Dataframe into Database
    Input:
        df: Dataframe to be saved
        database_filename: Pah of the DB to save the Dataframe
    Output:
        None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_Pipeline_Data', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()