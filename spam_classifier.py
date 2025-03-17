#Name: Brayden
#Project: Spam Email/SMS Classifier
#Date: 3/17/2025

# import needed dependencies
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# download necessary NLTK resources
nltk.download('punkt') # breaks up sentences into individual words
nltk.download('punkt_tab')
nltk.download('stopwords') #common english words

# lists all english stop words to filter out from the text such as "the", "is", "and"
stop_words = set(stopwords.words('english'))

# loads dataset from Kaggle of SMS spam and hams
df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']] #v1 is labels, v2 is messages
df.columns = ['label', 'text'] #rename columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1}) #converts labels into values

# displays basic dataset information
print("\nDataset Summary:")
print(f"Total messages: {len(df)}")
print(f"Spam messages: {df['label'].sum()}")
print(f"Ham messages: {len(df) - df['label'].sum()}\n")

# text cleaning function, removes punctation, lowercase text, and removes stop words
def clean_text(text):
    text = re.sub(r'\W+', ' ', text).lower().strip() #remove non word characters
    return ' '.join([word for word in word_tokenize(text) if word not in stop_words])

df['text'] = df['text'].apply(clean_text)

# convert text to number vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text']) # convert text to numerical format
y = df['label'] # targets variables

# splits data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# trains model
model = MultinomialNB()
model.fit(X_train, y_train)

# evaluate model test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) #calculate accuracy of the model

# find spam messages predicted by the model
spam_indices = [i for i in range(len(y_pred)) if y_pred[i] == 1]
original_indices = y_test.index[spam_indices]  # matches spam to original dataset

# extract text message that are shown to be spam
spam_messages = df.loc[original_indices, 'text'].values[:10]

# display results
print(f"Model Accuracy: {accuracy:.2f}")
print("Accuracy shows how often the model correctly identifies spam and ham messages.")
print("Higher accuracy means better spam detection, but some errors may still occur.")

# display sample spam messages
if len(spam_messages) > 0:
    print("\nList of Spam Messages Subjects:")
    for msg in spam_messages:
        print(f"- {msg}")
else:
    print("\nNo spam messages detected in this test sample.")
