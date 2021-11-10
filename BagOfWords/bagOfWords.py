#Bag of Words By Shane Cosgrove
#Appling NLP and DL to reviews using google Word2Vec

#Imports
import pandas as pd

#Read in File
train = pd.read_csv("/Users/Owner/KaggleComps/bagOfWords/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3);
#print(train.shape) #Prints out the number of rows,columns
#print(train.columns.values) #Print out the column names
#Print out the first review
#print(train["review"][0])
#print("-------")
#Data Cleaning and Text Preprocessing

#Import Date Cleaning package BeautifulSoup
from bs4 import BeautifulSoup

#Run BeautifulSoup over one review as an example
example1 = BeautifulSoup(train["review"][0],features="lxml")

#print(example1.get_text())

import re
#use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
#print(letters_only)

lower_case = letters_only.lower() #Convert to Lower case
words = lower_case.split() #split into individual words

#Removing Stop words
import nltk
nltk.download() #Dowload the stop words

from nltk.corpus import stopwords
print(stopwords.words("english"))

#Remove stop words from the data
words =[w for w in words if not w in stopwords.words("english")]

#Clean the rest of the reviews

def review_to_words(raw_review):#converts review to a string of words
    review_text = BeautifulSoup(raw_review).get_text() #Remove HTML
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) #Remove non letters
    letters_only.lower().split() #convert to lower case and split
    stops = set(stopwords.words("english")) #place stopwords into a set
    meaningful_words = [w for w in words if not w in stops] #remove stop words for reviews
    return(" ".join(meaningful_words)) #Join the words together and return them back

#Get the number of reviews
num_reviews = train["review"].size

#Create an empty list
clean_train_reviews=[]


print ("Cleaning and parsing the training set movie reviews...\n")
for i in range(0, num_reviews):
    #call function to clean reviews and add the result to the list
     if( (i+1)%1000 == 0 ):
            print ("Review %d of %d\n" % ( i+1, num_reviews ) )
            clean_train_reviews.append(review_to_words(train["review"][i]));
    
#Creating Features from a Bag of Words (Using scikit-learn)

print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print(train_data_features.shape)
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)

#Random Forest
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 
# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n") % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
    
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )