import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load datasets and merge them
    
    Input: messages dataset, categories dataset
    Output: Dataframe of both datasets merged
    
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,left_on='id',right_on='id')

    return df


def clean_data(df):
    
    """
    Transform to the correct format and clean dataframe for the model
    
    Input: Dataframe of both datasets merged
    Output: Dataframe processed and ready to be used by the model
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';',expand=True))

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # convert all values higher than 1 to 1 
        categories[column] = categories[column].apply(lambda x: 1 if x>=1 else 0)

    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # check number of duplicates
    df_duplicated = df.duplicated().sort_values(ascending=False)
    len_duplicated = len(df_duplicated[df_duplicated==True])

    if (len_duplicated>0):
        # drop duplicates
        df_no_duplicates = df.drop_duplicates()

        return df_no_duplicates
    else:
        print('There is not duplicated rows')

        return df



def save_data(df, database_filename):
    """
    Save dataframe in sql table
    
    Input: Dataframe cleaned, path to save sql table
    Output: Dataframe saved at the specified path
    
    """
    # 'sqlite:///MySQLdb.db'
    engine = create_engine(database_filename)
    df.to_sql('MySQL', engine, index=False)
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
