from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk.data

nltk.download()

def reviewToWords(rawReview):
    """
    Converts raw review to a string of words
    -Input is single html string
    -Output is preprocessed single string
    """
    cleanedReview = None
    
    #Remove HTML
    cleanedReview = BeautifulSoup(rawReview)
    
    #Remove numbers and punctuation
    cleanedReview = re.sub("[^a-zA-Z]",
                           " ",
                           cleanedReview.get_text())
    
    #Make all words lowercase
    cleanedReview = cleanedReview.lower()
    
    #Split into individual words
    cleanedReviewWords = cleanedReview.split()
    
    #Convert to set instead of list for efficiency
    stops = set(stopwords.words("english"))
    
    #Remove stop words
    meaningfulWords = [word for word in cleanedReviewWords if word not in stops]
    
    #Join words back into one string
    return (" ".join( meaningfulWords ))

def reviewToWordList(rawReview, removeStopWords = False):
    """
    Converts a document to sequence of words
    optionally removing stop words
    will later extend to optionally remove numbers
    
    I/O
    -Input: raw html in string form
    -Output: list of words
    """
    
    #Remove HTML
    cleanedReview = BeautifulSoup(rawReview).get_text()
    
    #Remove non-letters
    cleanedReview = re.sub("[^a-zA-Z]",
                           " ",
                           cleanedReview)
    
    #Convert words to lowerCase
    cleanedReview = cleanedReview.lower()
    
    #Split Words
    wordList = cleanedReview.split()
    
    #Optionally remove stop words
    if ( removeStopWords ):
        stops = set(stopwords.words('english'))
        wordList = [ word for word in wordList if word not in stops]
    
    #Return list of words
    return(wordList)
    
def reviewToSentences(rawReview, tokenizer, removeStopWords=False):
    """
    Converts review string into parsed sentences.
    Returns list of sentences --> each sentence is list of words
    """
    #Use NLTK tokenizer to split paragraph into sentences
    rawSentences = tokenizer.tokenize(rawReview.decode('utf-8').strip())
    
    #Loop over each sentence
    sentences = []
    for rawSentence in rawSentences:
        #skip empty sentences
        if len(rawSentence) > 0:
            #If non empty then call review to wordlist to get list of words for that sentence
            sentences.append( reviewToWordList(rawSentence, removeStopWords))
    
    #return list of sentences which are themselves list of words
    return sentences

def makeFeatureVector(words, model, numberFeatures):
    """
    Averages all of the word vectors in a given paragraph
    """
    
    #Pre-initialize empty numpy array (for speed)
    featureVec = np.zeros((numberFeatures), dtype='float32')
    
    numberWords = 0
    
    #Index2Word is list containing the names of the words the models vocab
    #We convert it to a set for speed
    indexWordSet = set(model.index2word)
    
    #Loop over each word in review
    # if word in models vocab
    #   add its feature vector to the total
    for word in words:
        if word in indexWordSet:
            numberWords = numberWords + 1
            featureVec = np.add(featureVec, model[word])
            
    #Divide result by the number of words to get the average
    featureVec = np.divide(featureVec, numberWords)
    return featureVec

def getAvgFeatureVectors(reviews, model, numberFeatures):
    """
    Given set of reviews (each one list of words),
    calculates the avg feature vector for each one
    and returns a 2D numpy array
    """
    
    #Initialize counter
    counter = 0
    
    #Preallocate 2D numpy array for speed
    reviewFeatureVectors = np.zeros((len(reviews), numberFeatures), dtype='float32')
    
    #Loop through the reviews
    for review in reviews:
        #print status message every 100th review
        if ( (counter + 1) % 1000 == 0 ):
            print 'making average feature vector for review %d of review %d' % (counter, len(reviews))
            
        #Make average feature vector for a given review
        reviewFeatureVectors[countr] = makeFeatureVector(review, model, numberFeatures)
        counter = counter + 1
    
    #return all the averaged feature vectors
    return reviewFeatureVectors


def createBagOfCentroids(wordList, wordCentroidMap):
    """
    Function will give us numpy array for each review
    Each one will have number of features equal to number of clusters
    """
    #
    #Number of centroids is equal to the highest cluster index in the word/centroid map
    numCentroids = max(wordCentroidMap.values()) + 1
    
    #Preallocate bag of centroid vectors (for speed)
    bagOfCentroids = np.zeros(numberCentroids, dtype='float32')
    
    #Loop over words in review. If word in vocab then find which cluster it belongs to
    #and increment clustercount by 1
    for word in wordList:
        if word in wordCentroidMap:
            index = wordCentroidMap[word]
            bagOfCentroids[index] +=1
    
    return bagOfCentroids
    
    
    
        
    
    
    
 
    
 