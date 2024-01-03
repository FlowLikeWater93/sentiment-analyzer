import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helperFunctions as hf

'''
The data is stored in a quite large file (5GB)
pd.read_json() command was not used because the file is too large it would cause most computers to freeze as it loads the data into RAM
In order to work around this issue :
1- open the json as file and go through it one line at a time
2- Read and analyze 1 million json objects at once
3- Save a few columns as a dataframe. The large size comes from the text column
'''

def find_stats():
    '''
    uses the helper helper function to load the data
    from the large json file without the text

    Return :
    dataframe with full data of only three columns
    - Stars
    - business_id
    - a custom column with the number of words per review
    - review date
    '''
    analysis_df = None

    # for loop to go through the whole json file
    # one million reviews per iteration
    for i in range(7):
        print((i+1), '- Loading ...')

        # Load 1 million reviews starting from offset
        df = hf.handle_large_jsons(1000000, (i*1000000))

        # make sure date column has the correct format
        df['date'] = pd.to_datetime(df['date'])
        # make sure stars column is of integer type
        df['stars'] = df['stars'].astype(int)
        # create a custom column with number of words in each review
        df['review_size'] = df['text'].apply(lambda x: len(x.split(' ')))
        # We are going to maintain the actual data only for the columns 2, 3, 8 and 9
        if i == 0:
            analysis_df = df.iloc[:, [2, 3, 8, 9]].copy()
        else :
            analysis_df = pd.concat([analysis_df, df.iloc[:, [2, 3, 8, 9]].copy()])

    return analysis_df

# Run the function and save the retunred objects
analysis_df = find_stats()

# Dislay the results
print('\n')
print('1- review date')
print('Max date : {} - Min date : {}'.format(analysis_df['date'].max(), analysis_df['date'].min()))
print('\n')
print('2- Star ratings')
print('Mean star rating : {} stars'.format(round(analysis_df.stars.mean(),2)))
print('Percentage of times a star rating was given : ')
print(analysis_df.stars.value_counts()/analysis_df.shape[0])
print('\n')
print('3- Text review length')
print(analysis_df.review_size.describe())
print('\n')
print('4- reviewed business')
print(analysis_df.business_id.value_counts().reset_index(name='count')['count'].describe())


print('\nGenerating Visualizations ...')

# uncommnet each block seperatly to save individual plots
# plt.rc('figure', figsize=(10, 8))

# block 1
# bins = np.arange(0,3000,50)
# plt.hist(analysis_df['review_size'], bins=bins)
# plt.xlabel("Number of words")
# plt.ylabel("Reviews (in millions)")
# plt.savefig('graphs/word_counts.png')

# block 2
# bins = np.arange(0,1000,50)
# plt.hist(analysis_df.business_id.value_counts().reset_index(name='count').query('count <= 1000')['count'], bins=bins, color='skyblue')
# plt.xlabel("Number of reviews")
# plt.ylabel("Businesses")
# plt.savefig('graphs/business_reviews.png')

# block 3
#plot_df['stars_2'] = plot_df['stars'].apply(lambda x : str(x)+' stars' if x >1 else str(x)+' star')
#plt.pie(plot_df['count'], labels = plot_df['stars_2'], colors=["#A9A9A9", "hotpink", "b", "#4CAF50", "#F8F8FF"])
#plt.savefig('graphs/stars_share.png')
