import pandas as pd
import numpy as np
import hazm
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import math

normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()
translator = str.maketrans('', '', string.punctuation)
sw = hazm.stopwords_list()

def preprocess_text(text):
    normalized_text = normalizer.normalize(text)
    punct_normalized = re.sub(r'[^\w\s]', '', normalized_text)
    tokens = hazm.word_tokenize(punct_normalized)
    tokens_without_stopwords = [token for token in tokens if token not in sw]
    tokens_without_numbers = [re.sub(r'\d+', '', token) for token in tokens_without_stopwords]
    non_empty_tokens = [token for token in tokens_without_numbers if token.strip() != '']
    # stemmed_tokens = [stemmer.stem(token) for token in non_empty_tokens]
    return ' '.join(non_empty_tokens)


train_df = pd.read_csv("books_train.csv")
test_df = pd.read_csv("books_test.csv")
train_df['processed_text'] = train_df['title'].astype(str) + ' ' + train_df['description'].astype(str)
train_df['processed_text'] = train_df['processed_text'].apply(preprocess_text)

test_df['processed_text'] = test_df['title'].astype(str) + ' ' + test_df['description'].astype(str)
test_df['processed_text'] = test_df['processed_text'].apply(preprocess_text)

grouped_df = train_df.groupby('categories')['processed_text'].apply(' '.join).reset_index()
vectorizer = CountVectorizer()
category_matrix = vectorizer.fit_transform(grouped_df['processed_text'])
BOW = pd.DataFrame(category_matrix.toarray(), columns=vectorizer.get_feature_names_out())
BOW['categories'] = grouped_df['categories']
BOW = BOW.groupby('categories').sum().reset_index()

bow = BOW.drop('categories', axis=1).values
row_sums = np.sum(bow, axis=1)
normalized_bow = bow / row_sums[:, np.newaxis]
normalized_BOW = pd.DataFrame(normalized_bow, columns=BOW.columns[:-1])
normalized_BOW['categories'] = BOW['categories']
# now we have  p(x|c) for each unique word


results = []
for index, row in test_df.iterrows():
    text = row['processed_text']
    category_scores = []
    for category_index in range(len(normalized_BOW)):
        word_scores = []

        for word in text.split():
            if word in normalized_BOW.columns:
                word_score = normalized_BOW.loc[category_index, word]
                if word_score == 0:
                    word_score = 0.0000000000001
                word_scores.append(word_score)
            else:
                word_scores.append(0.0000000000001)

        category_score = np.sum(np.log(word_scores))
        # category_score = np.prod(word_scores)
        category_scores.append(category_score)

    max_score_index = np.argmax(category_scores)
    predicted_category = normalized_BOW.loc[max_score_index, 'categories']
    results.append(predicted_category)

unique_results = set(results)
if len(unique_results) == 1:
    results = [-1] * len(results)

test_df['predicted_category'] = results
test_df['predicted_category'].to_csv('prediction_8.csv', index=False)
test_df['categories'].to_csv('categories_p1.csv', index=False)


import csv

def compare_csv_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        results = []
        for row1, row2 in zip(reader1, reader2):
            results.append(row1 == row2)

        return results


result = compare_csv_files('prediction_8.csv', 'categories_p1.csv')

true_count = result.count(True)
total_count = len(result)
accuracy = (true_count/total_count)*100
print("correct predictions : " + str(true_count))
print("total : " + str(total_count))
print("accuracy percentage : " + str(accuracy)[:4])



