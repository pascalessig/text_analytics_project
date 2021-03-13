import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(path_to_apps, path_to_reviews, amount_reviews):
    # loading in reviews
    reviews_df = pd.read_csv(path_to_reviews, nrows = amount_reviews)
    apps_df = pd.read_csv(path_to_apps)
    #reduce metadata to whats important for analysis
    apps_df = apps_df[['app_id', 'title', 'genre', 'rating', 'comp_name', 'description', 'downloads']]
    return apps_df, reviews_df

def get_comp_names():
    comps = list(apps_df['comp_name'])
    return [str(c).lower() for c in comp]

def lemma(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

def remove_stopw(text, stop_words):
    stopped = [word for word in text if (word not in stop_words)]
    return [word for word in stopped if (len(word) > 3)]

def preprocess_text(text):
    text = text.apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)).lower())
    text = text.apply(lambda x: word_tokenize(str(x)))
    text = text.apply(lemma)
    return text
    
def is_in(x, keep):
    return (x in keep )

def get_processed_data(path_to_apps = '../data.nosync/app_data_exam_09052020.csv', path_to_reviews = '../data.nosync/reviews_09052020.csv', amount_reviews = 50000, keep_category = ['Art & Design', 'Augmented Reality', 'Auto & Vehicles', 'Beauty', 'Books & Reference', 'Business', 'Comics', 'Communication', 'Dating', 'Daydream', 'Education', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle', 'Maps & Navigation']):
    print('loading files...')
    apps_df, reviews_df = load_data(path_to_apps, path_to_reviews, amount_reviews)
    print('slice categories to "%s"...' %keep_category)
    keep_bool = apps_df['genre'].apply(lambda x: is_in(x,keep_category))
    apps_df = apps_df[keep_bool]
    cat_id = str(apps_df.iloc[0,0])[:5]
    reviews_df = reviews_df[reviews_df['rev_id'].apply(lambda x: x[:5] == cat_id)]
    print('preprocess metadata...')
    apps_df = apps_df[apps_df['rating'] > 0]
    print(apps_df.shape)
    apps_df["description_normalized"] = preprocess_text(apps_df["description"])
    comp_names = list(apps_df['comp_name'])
    stop_words = stopwords.words('english')
    extension = ['drpandagam', 'dr', 'panda', 'drpanda', 'http'] + comp_names
    stop_words = list(stop_words) + extension
    apps_df["description_normalized"] = apps_df["description_normalized"].apply(lambda x: remove_stopw(x, stop_words))
    print('preprocess reviews...')
    reviews_df["text_normalized"] = preprocess_text(reviews_df["text"])
    print('I remove duplicates in apps_df and you in reviews_df deal?')
    print('Removing in apps_df...')
    sum_dups = apps_df[apps_df['description'].duplicated(keep='first')].shape[0]
    print('Found %i duplicates.' %sum_dups)
    redun_index = list(apps_df[apps_df.description.duplicated(keep='first')].index)
    apps_df.drop(redun_index, axis=0, inplace=True)
    print('DONE!')
    return apps_df, reviews_df
