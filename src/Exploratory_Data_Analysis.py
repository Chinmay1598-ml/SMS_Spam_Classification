# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import nltk
# import string
# import re
# from wordcloud import WordCloud
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Loading dataset

df = pd.read_csv(r"C:\Users\chinm\Downloads\Message_Classification.csv")
print(df)

print(df.info())
print(df.describe())

print(df.shape)
print(df.isnull().sum())
print(df['Sender'].value_counts())

df['Sender'] = df['Sender'].map({'ham': 0, 'spam': 1})
print(df)


sns.countplot(data=df, x='Sender', palette='pastel')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title("Distribution of Ham vs Spam Messages")
plt.show()

# Add a feature for message length
df['message_length'] = df['Messages'].apply(len)

# Plot message length distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='message_length', hue='Sender', kde=True, bins=30, palette='coolwarm')
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# Wordcloud for spam messages
spam_words = ' '.join(df[df['Sender'] == 1]['Messages'])
spam_wc = WordCloud(width=800, height=400, background_color='black').generate(spam_words)

# Plot spam wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(spam_wc, interpolation='bilinear')
plt.title("Spam Wordcloud")
plt.axis('off')
plt.show()

# Wordcloud for ham messages
ham_words = ' '.join(df[df['Sender'] == 0]['Messages'])
ham_wc = WordCloud(width=800, height=400, background_color='white').generate(ham_words)

# Plot ham wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(ham_wc, interpolation='bilinear')
plt.title("Ham Wordcloud")
plt.axis('off')
plt.show()

# Average message length per class
print(df.groupby('Sender')['message_length'].mean())

# Common words in spam vs ham

vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['Messages'])

# Convert to DataFrame
top_words_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Spam vs Ham Word Counts
spam_words = top_words_df[df['Sender'] == 1].sum().sort_values(ascending=False)
ham_words = top_words_df[df['Sender'] == 0].sum().sort_values(ascending=False)

print("Top Words in Spam Messages:\n", spam_words)
print("\nTop Words in Ham Messages:\n", ham_words)

# Average message length per class
print(df.groupby('Sender')['message_length'].mean())

# Common words in spam vs ham

vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['Messages'])

# Convert to DataFrame
top_words_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Spam vs Ham Word Counts
spam_words = top_words_df[df['Sender'] == 1].sum().sort_values(ascending=False)
ham_words = top_words_df[df['Sender'] == 0].sum().sort_values(ascending=False)

print("Top Words in Spam Messages:\n", spam_words)
print("\nTop Words in Ham Messages:\n", ham_words)

# Save the processed dataset for modeling
df.to_csv("C:/Users/chinm/Downloads/processed_spam.csv", index=False)
