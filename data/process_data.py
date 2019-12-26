import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Input:
        messages_filepath -> path to the file containing messages
        categories_filepath -> path to the file containing categories and labels
            
    Output: 
        df -> a single data frame containing messages and respective categories/labels
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, sep= ';', delimiter = ',')
    df = pd.merge(messages, categories, on = 'id', how = 'left')
    
    return df


def clean_data(df):
    """
    Input: 
        df -> data frame containing uncleaned category column
    
    Output:
        df -> with cleaned categories columns
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df.categories).str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = [x.split('-')[0] for x in row]
    #  rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [x.split('-')[1] for x in categories[column]]
    
    # convert column from string to numeric
        categories[column] = [int(x) for x in categories[column]]
        
    # drop the original categories column from `df`
    df.drop(columns = 'categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    return df


def save_data(df, database_filename):
    # create sqlite engine
    engine = create_engine('sqlite:///' + str(database_filename))

    # send df to sqlite file, omitting the index
    df.to_sql('disaster_data', engine, index=False)


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