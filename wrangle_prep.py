import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

import env
from env import user, password, host
import warnings
warnings.filterwarnings('ignore')




def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    '''
    
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df




def wrangle_zillow():
    ''' 
    '''

    filename = 'zillow.csv'
    
    if os.path.exists(filename):
        print('Reading cleaned data from csv file...')
        return pd.read_csv(filename)
    
    
    query = '''
        SELECT prop.*, 
               pred.logerror, 
               pred.transactiondate, 
               air.airconditioningdesc, 
               arch.architecturalstyledesc, 
               build.buildingclassdesc, 
               heat.heatingorsystemdesc, 
               landuse.propertylandusedesc, 
               story.storydesc, 
               construct.typeconstructiondesc 
               
        FROM properties_2017 prop  
                INNER JOIN (SELECT parcelid,
                                  logerror,
                                  Max(transactiondate) transactiondate 
        FROM predictions_2017 
                GROUP BY parcelid, logerror) pred USING (parcelid)
                
        LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
        LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
        LEFT JOIN storytype story USING (storytypeid) 
        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
        
        WHERE prop.latitude IS NOT NULL 
        AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
        '''

    url = f"mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow"

    df = pd.read_sql(query, url)
    
    # Single units only
    single_unit = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    
    # Refine
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.unitcnt<=1)|df.unitcnt.isnull()]
    
    # Missing Values
    df = handle_missing_values(df)
    
    # Columns to drop
    columns_to_drop = ['id', 'heatingorsystemdesc', 'heatingorsystemtypeid', 'finishedsquarefeet12', 'calculatedbathnbr', 'propertycountylandusecode', 'censustractandblock', 'fullbathcnt', 'propertylandusetypeid', 'propertylandusedesc', 'propertyzoningdesc', 'unitcnt', 'transactiondate']
    df = df.drop(columns=columns_to_drop)
    
    # Remove nulls for buildingqualitytypeid and lotsizesquarefeet
    df.buildingqualitytypeid.fillna(6.0, inplace= True)
    df.lotsizesquarefeet.fillna(7313, inplace=True)
    
    # Remaining nulls
    df.dropna(inplace=True)
    
    # Outliers
    df = df[df.calculatedfinishedsquarefeet < 9000]
    df = df[df.taxamount < 20000]
    
    # Download cleaned data to a .csv
    df.to_csv(filename, index=False)
    
    print('Downloading data from SQL...')
    print('Saving to .csv')
    return df


def split_data(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=1313)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=1313)

    # Take a look at your split datasets

    print(f'train <> {train.shape}')
    print(f'validate <> {validate.shape}')
    print(f'test <> {test.shape}')
    return train, validate, test



def scale_data(train, validate, test, return_scaler=False):
    '''
    This function takes in train, validate, and test dataframes and returns a scaled copy of each.
    If return_scaler=True, the scaler object will be returned as well
    '''
    
    scaler = MinMaxScaler()
    
    num_columns = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxamount', 'roomcnt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler.fit(train[num_columns])
    
    train_scaled[num_columns] = scaler.transform(train[num_columns])
    validate_scaled[num_columns] = scaler.transform(validate[num_columns])
    test_scaled[num_columns] = scaler.transform(test[num_columns])
    
    print(f'train_scaled <> {train_scaled.shape}')
    print(f'validate_scaled <> {validate_scaled.shape}')
    print(f'test_scaled <> {test_scaled.shape}')
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    
# Outliers
# Borrowed from Zac

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.
    
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))



def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
    #                 for col in df.select_dtypes('number')}
    # return df.assign(**outlier_cols)
    
    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)
        
    return df
    
    
# Functions for null metrics

def column_nulls(df):
    missing = df.isnull().sum()
    rows = df.shape[0]
    missing_percent = missing / rows
    cols_missing = pd.DataFrame({'missing_count': missing, 'missing_percent': missing_percent})
    return cols_missing



def columns_missing(df):
    df2 = pd.DataFrame(df.isnull().sum(axis =1), columns = ['num_cols_missing']).reset_index()\
    .groupby('num_cols_missing').count().reset_index().\
    rename(columns = {'index': 'num_rows' })
    df2['pct_cols_missing'] = df2.num_cols_missing/df.shape[1]
    return df2



# Missing Values

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df




def plot_categorical_and_continuous_vars(x, y, df):
    '''
    This function accepts your dataframe and the name of the columns that hold 
    the continuous and categorical features and outputs 3 different plots 
    for visualizing a categorical variable and a continuous variable.
    '''
    
    # Title
    plt.suptitle(f'{x} by {y}')
                 
    # Lineplot
    sns.lineplot(x, y, data=df)
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Swarm Plot
    sns.catplot(x, y, data=df, kind='swarm', palette='Greens')
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Box Plot
    sns.catplot(x, y, data=df, kind='box', palette='Blues')
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Bar Plot
    sns.catplot(x, y, data=df, kind='bar', palette='Purples')
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Scatter plot with regression line
    sns.lmplot(x, y, data=df)
    plt.xlabel(x)
    plt.ylabel(y)
    
    plt.show()

    return train, validate, test






def plot_variable_pairs(train, columns, hue=None):
    '''
    The function takes in a df and a list of columns from the df
    and displays a pair plot wid a regression line.
    '''
    
    kws = {'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}}
    sns.pairplot(train[columns],  kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
    plt.show()
    
    
    
    
def rfe_feature_rankings(x_scaled, x, y, k):
    '''
    Takes in the predictors, the target, and the number of features to select,
    and it should return a database of the features ranked by importance
    '''
    
    # Make it
    lm = sklearn.linear_model.LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n_features_to_select=k)

    # Fit it
    rfe.fit(x_scaled, y)
    
    var_ranks = rfe.ranking_
    var_names = x.columns.tolist()
    ranks = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    ranks = ranks.sort_values(by="Rank", ascending=True)
    return ranks
    
# *