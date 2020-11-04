import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function
    Arguments:
    messages_filepath -> path to messages csv file
    categories_filepath -> path to categories csv file
    Output:
    df -> Loaded dasa as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer',on=['id'])  
    return df


def clean_data(df):
    """
    Clean Data function
    Arguments:
    df -> raw data Pandas DataFrame
    Outputs:
    df -> clean data Pandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.concat([df['categories'].str.split( ";",expand=True)], axis=1)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
        # drop the original categories column from `df`
        df.drop(['categories'], axis=1, inplace=True)
        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.concat([df,categories], join='inner', axis=1)
        # drop duplicates
        df.drop_duplicates(inplace=True)
        return df


def save_data(df, database_filename):
    """
    Save Data function
    Arguments:
    df -> Clean data Pandas DataFrame
    database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass

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