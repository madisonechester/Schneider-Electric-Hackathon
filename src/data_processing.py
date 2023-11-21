import os 
import pandas as pd
import numpy as np

def fill_area_code(df):

    df['AreaID'].dropna(inplace=True)
    cn_id = df['AreaID'][0]
    #print(cn_id)
    df['AreaID'] = cn_id
    
    return df
        
def read_and_concatenate(folder_path):
    # Lists to store DataFrames
    gen_dataframes = []
    load_dataframes = []

    # Iterate over all files in the folder
    list_files = [file for file in os.listdir(folder_path) if file!='test.csv']
    for file in list_files:
        if file.endswith('.csv'):
            # print('---------------------------------')
            # print(file)
            file_path = os.path.join(folder_path, file)
            

            # Read 'gen' files
            if file.startswith('gen'):
                df = pd.read_csv(file_path)
                if file == 'gen_SP_B10.csv':
                    df['AreaID'] = '10YES-REE------0'
                df = fill_area_code(df)
                df['quantity'].fillna(0, inplace=True)
                gen_dataframes.append(df)


            # Read 'load' files
            elif file.startswith('load'):
                df = pd.read_csv(file_path)
                df = fill_area_code(df)
                df['Load'].fillna(0, inplace=True)
                load_dataframes.append(df)

    print('=====================================')
    print('Read all files in the raw_data folder')   
    print('=====================================')

    # Concatenate DataFrames vertically
    gen_concatenated = pd.concat(gen_dataframes, axis=0, ignore_index=True)
    load_concatenated = pd.concat(load_dataframes, axis=0, ignore_index=True)

    # Combine 'gen' and 'load' DataFrames
    combined_dataframe = pd.concat([gen_concatenated, load_concatenated], axis=0, ignore_index=True)

    print('=====================================')
    print('Combined all data frames')   
    print('=====================================')

    return combined_dataframe

def further_processing(df):
    print('=====================================')
    print('Further processing')   
    print('=====================================')

    # change date format
    df['StartTime'] = pd.to_datetime(df['StartTime'].str.replace('\+00:00Z', '', regex=True)).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['EndTime'] = pd.to_datetime(df['EndTime'].str.replace('\+00:00Z', '', regex=True)).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df['EndTime'] = pd.to_datetime(df['EndTime'])

    df['AreaID'] = df['AreaID'].replace({
        '10YHU-MAVIR----U': 'HU',
        '10YIT-GRTN-----B': 'IT',
        '10YPL-AREA-----S': 'PO',
        '10YES-REE------0': 'SP',
        '10Y1001A1001A92E': 'UK',
        '10Y1001A1001A83F': 'DE',
        '10Y1001A1001A65H': 'DK',
        '10YSE-1--------K': 'SE',
        '10YNL----------L': 'NE'
    })

    df.fillna(0, inplace=True)
    df['gen/load'] = 'load'
    df['Load'] = df['Load'].fillna(0)
    df.loc[df['Load']==0,'gen/load']='gen'
    df['power'] = df['quantity'] + df['Load']
    
    # Extract date and hour
    df['Date'] = df['StartTime'].dt.date
    df['Hour'] = df['StartTime'].dt.hour

    # aggregate data per country, load/gen, date and hour
    aggregated_data = df.groupby(['AreaID', 'gen/load', 'Date', 'Hour'])['power'].sum().reset_index()
    aggregated_data['concatenated'] = aggregated_data['AreaID']  + aggregated_data['gen/load']


    pivot = aggregated_data.pivot_table(
        index=['Date', 'Hour'],
        columns=['concatenated'],
        values='power',
        aggfunc='sum'
    )


    pivot.replace(0, np.nan, inplace=True)
    # Drop rows where all elements are NaN
    pivot.dropna(how='all', inplace=True)
    pivot = pivot.reset_index()



    pivot['Date'] = pd.to_datetime(pivot['Date'])
    pivot = pivot[pivot['Date'].dt.year == 2022]

    #Add surpluses columns
    for country in ['HU','IT','PO','SP','DE','DK','SE','NE']:
        pivot[country+'_surplus']=pivot[country+'gen']-pivot[country+'load']


    # Map column names to numbers
    labels_countries = {
        'SP_surplus': 0, # Spain
        'UK_surplus': 1, # United Kingdom
        'DE_surplus': 2, # Germany
        'DK_surplus': 3, # Denmark
        'HU_surplus': 5, # Hungary
        'SE_surplus': 4, # Sweden
        'IT_surplus': 6, # Italy
        'PO_surplus': 7, # Poland
        'NL_surplus': 8 # Netherlands
    }

    
    # Find the column with the maximum value for each row and map it to the corresponding number
    pivot['label'] = pivot[['HU_surplus', 'IT_surplus', 'PO_surplus', 'SP_surplus', 
                    'DE_surplus', 'DK_surplus', 'SE_surplus', 'NE_surplus']].idxmax(axis=1).map(labels_countries)


    # Define European dates for seasons
    spring_start = pd.to_datetime('2022-03-21')
    summer_start = pd.to_datetime('2022-06-21')
    autumn_start = pd.to_datetime('2022-09-22')
    winter_start = pd.to_datetime('2022-12-21')

    # Create a new column 'season' based on the defined seasons
    pivot['season'] = 'winter'  # Default to winter

    # Set conditions for other seasons
    spring_condition = (pivot['Date'] >= spring_start) & (pivot['Date'] < summer_start)
    summer_condition = (pivot['Date'] >= summer_start) & (pivot['Date'] < autumn_start)
    autumn_condition = (pivot['Date'] >= autumn_start) & (pivot['Date'] < winter_start)

    # Update 'season' based on conditions
    pivot.loc[spring_condition, 'season'] = 'spring'
    pivot.loc[summer_condition, 'season'] = 'summer'
    pivot.loc[autumn_condition, 'season'] = 'autumn'

    pivot['season'] = pivot['season'].astype(str)

    # One-hot encode the 'season' column
    season_dummies = pd.get_dummies(pivot['season'], drop_first=True)

    # Concatenate one-hot encoded columns to the DataFrame
    pivot = pd.concat([pivot, season_dummies], axis=1)

    # Drop the original 'season' column
    pivot.drop(columns=['season'], inplace=True)

    season_columns = ['spring', 'summer', 'winter']
    pivot[season_columns] = pivot[season_columns].astype(int)

    # Extract day of the week and create 'weekend' column, replacing True/False with 1/0
    pivot['day_of_week'] = pivot['Date'].dt.dayofweek
    pivot['is_weekend'] = pivot['day_of_week'].isin([5, 6]).astype(int)

    # Extract 'Country' feature from the 'concatenated' column
    #pivot['Country'] = pivot['concatenated'].str[:-4]


    # handle missing data
    # pivot.interpolate(method='linear', limit_direction='both', inplace=True)

    pivot.to_csv('../data/final_data.csv', index=False)
    print('=====================================')
    print('Pivot df saved to final_data')   
    print('=====================================')
    
    return pivot

def main():
    folder_path = '../data/raw_data/'
    data = read_and_concatenate(folder_path)
    pivot = further_processing(data)

if __name__ == "__main__":
    main()