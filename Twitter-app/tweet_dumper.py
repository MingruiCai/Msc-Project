#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import csv
import re
import snowballstemmer

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from nltk.probability import FreqDist

from geotext import GeoText

# Twitter API credentials
consumer_key = "tndAcVLSnfdGVxHXSrtCx7yY0"
consumer_secret = "dXZD2WAHW6vfQIo1L77HLGp9SmvETn058sMaXaQoh9Zm7HcQD3"
access_key = "1405286036473843714-QVmhQZVHpSPzICwqjXokxVoSTRslb5"
access_secret = "7gBfupUaKCM5boYQYJK1YYDEvpILhTTIWr3ESXGnS7gOQ"
#  remove stop words
stopwords = []
file = "stopwords.txt"
with open(file, "r") as f1:
    for stop_word in f1.readlines():
        if stop_word is not None:  # read stop words from a txt file and store then into a list.
            stopwords.append(stop_word.strip("\n"))


def init(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []
    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode="extended")
    # save most recent tweets
    alltweets.extend(new_tweets)
    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, tweet_mode="extended")
        # save most recent tweets
        alltweets.extend(new_tweets)
        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        outtweets = [tweet.full_text for tweet in alltweets]
    return alltweets, outtweets


def getCityname(alltweets):
    # extract the place name from the timeline
    placeset = set()
    for tweet in alltweets:
        if tweet.place is not None and tweet.place.place_type == "city":
            # print(tweet.place.place_type)
            placeset.add(tweet.place.name)
    placeslist = list(placeset)
    return placeslist


def getHashtags(alltweets):
    hashtags = []
    for tweet in alltweets:
        hashtaginfo = tweet.entities.get("hashtags")
        if len(hashtaginfo) != 0:
            hashtags.append(hashtaginfo[0].get("text").lower())
    tag_freq = FreqDist(hashtags)
    # print(tag_freq.most_common())
    # filter out the tags with high frequency
    tag2class = [tag[0] for tag in tag_freq.most_common() if tag[1] >= 5]  # high possibility
    return tag2class


# preprocess used for builidng inverted index including tokenisation, casefolding,
# removing stop words and porter stemming.
def preprocess(text):
    remove_url = re.sub(r"http\S+", "", text)
    # split non-letter characters and store them into a list.
    regEx = re.compile("\W").split(remove_url)
    token = [i for i in regEx if i != '']
    # all in lower case.
    casefolding = [s.lower() for s in token if isinstance(s, str) == True]
    # removing stop words.
    stopwords_removed = [w for w in casefolding if not w in stopwords]
    # Porter stemmer by using snowball stemmer lib.
    stemmer = snowballstemmer.stemmer('english')
    return stemmer.stemWords(stopwords_removed)


def getTweetsout(alltweets):
    outtweets = [tweet.full_text for tweet in alltweets]
    preprocessed_tweets = []
    for tweet in outtweets:
        preprocessed_tweets.append(preprocess(tweet))
    return preprocessed_tweets


def LDA(preprocessed_tweets):
    # word frequency
    # flat_words = [item for sublist in preprocessed_tweets for item in sublist]
    # print(flat_words)
    # word_freq = FreqDist(flat_words)

    # create dictionary
    text_dict = Dictionary(preprocessed_tweets)
    tweets_bow = [text_dict.doc2bow(tweet) for tweet in preprocessed_tweets]
    # fitting lda model
    tweets_lda = LdaModel(tweets_bow, num_topics=10, id2word=text_dict, random_state=1)

    x = tweets_lda.show_topics()
    alltopicWords = []
    for topicNum, topicWords in x:
        words_foreachtopic = re.findall("[a-zA-Z]+", x[topicNum][1])
        alltopicWords += words_foreachtopic
    alltopicWordsset = set(alltopicWords)
    alltopicWords = list(alltopicWordsset)
    # print(alltopicWords)
    return alltopicWords, text_dict,tweets_bow


def core_algorithm(newtweet, tag2class, alltopicWords, outtweets,text_dict, tweets_bow):
    # 1. recognize the city name in the new tweet

    # Determine if a city name exists because this python library only supports city name recognition
    # but the tweet place entity can have place type other than city such as admin
    placename = GeoText(newtweet).cities
    if len(placename) != 0 and set(placename) <= set(placeslist):
        return "This new tweet is fine to post ! place"
    # 2a. get hashtags from the new tweet
    # initializing hashtag_list variable
    hashtag_list = []
    # splitting the text into words
    for word in newtweet.lower().split():

        # checking the first character of every word
        if word[0] == '#':
            if word[1:] != "":
                # adding the word to the hashtag_list
                hashtag_list.append(word[1:].lower())
    # 2b. If all the hashtags of new tweet has appeared in previous tweets, then we decide to pass it
    # print(hashtag_list)
    # print(tag2class)
    if len(hashtag_list) != 0 and set(hashtag_list) <= set(tag2class):
        return "This tweet is fine to post! hashtag"

    # 3. Determine if the number of key tokens in new tweets
    #    is over the threshold
    txt = preprocess(newtweet)
    counter = 0
    for token in txt:
        if token in alltopicWords:
            counter += 1
            print(token)
    if counter >= round(len(alltopicWords) / 10):
        return "This tweet is fine to post! topics word"

    # 4. using lsi to measure the similarity of new tweet and old corpus
    # if there are previous tweets that get over 0.97 score, then we can say the new tweet is similar to some previous tweets
    corpus = tweets_bow
    id2word = text_dict
    dictionary = text_dict
    from gensim import models
    lsi = models.LsiModel(corpus=tweets_bow, id2word=text_dict, num_topics=10)

    doc=newtweet
    vec_bow = dictionary.doc2bow(preprocess(doc))
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space

    from gensim import similarities
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('deerwester.index')
    index = similarities.MatrixSimilarity.load('deerwester.index')
    # perform a similarity query against the corpus
    sims = index[vec_lsi]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for doc_position, doc_score in sims:
        if doc_score >= 0.97:
            return "This tweet is fine to post! similarity"
            # print(doc_score, outtweets[doc_position])

if __name__ == '__main__':
    # pass in the username of the account you want to download
    alltweets,outtweets = init("EdinburghUni")
    placeslist = getCityname(alltweets)
    tag2class = getHashtags(alltweets)
    # print(outtweets)
    # del (outtweets[0])
    # print(outtweets)
    preprocessed_tweets = getTweetsout(alltweets)
    alltopicWords, text_dict, tweets_bow = LDA(preprocessed_tweets)

    newtweet = " Banning in-car smoking when children are present can reduce their exposure to tobacco smoke by more than 30 per cent, new research from @EdinUniUsher shows."
    suggestion = core_algorithm(newtweet, tag2class, alltopicWords, outtweets,text_dict, tweets_bow)
    if suggestion==None:
        print("This tweet may involve some sensitive/uncommon information for you !")
        print("Are you sure you want to post it?")
    else:
        print("This tweet is totally fine to post!")
