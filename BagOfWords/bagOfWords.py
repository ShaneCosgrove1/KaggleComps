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
#nltk.download() #Dowload the stop words

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
    
