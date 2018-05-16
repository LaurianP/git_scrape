# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:36:22 2018

@author: popaz
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import WordNetLemmatizer


def consistentDateTime(df, timeColName):
    #utility to apply the same standard treatment to datetimes, e.g.
    #forcecast the column from obj type into datetime stamp and
    #eliminate the timestamp past, since that will defeat joins and groupby. 
    
    # some rows are empty. if the date is empty, the row is empty. 
    df = df[pd.notnull(df[timeColName])]
    df[timeColName] = pd.to_datetime(df[timeColName])
    
    df[timeColName] = df[timeColName].dt.normalize() # removes the timestamp
    return(df)
    
def collapseDateDuplicates(df):
    # only applies to the 'news' data frame. the returns won't have duplicates. 
    tmpgroup = df.groupby(["date", "searchtext", "source"])
    df = tmpgroup.agg({'text':' '.join, 'title': ' '.join}).reset_index() #concatenates strings
    return(df)
    

def loadCleanCsv(csvName):
    df = pd.read_csv(csvName)
    # Crucial: the Date field should be datetime not object (string)
    # This will facilitate subsequent joins and queries. 
    df = consistentDateTime(df, 'date')

    ### clean up the 'text' field ###
    punctRegexp = re.compile(r'[\s:-]+') 
    # replace some separators with spaces
    df['text'] = df['text'].apply(lambda x: punctRegexp.sub(' ', x))
    # drop any non-ascii encodings (e.g. unicode characters)
    df['text'] = df['text'].apply(lambda x: x.encode('ascii', errors='ignore').decode())
    # 22.75PER SHARE should be 22.75 PER SHARE
    # eliminate dot as punctuation, not inside decimals
    df['text'] = df['text'].apply(lambda x: re.sub('(?<=\D)[.,]|[.,](?=\D)', ' ', x))
    # remove stop words
    NLTKstopWords = set(stopwords.words('english'))
    removeStops = lambda x: ' '.join([word for word in x.split() if word.lower() not in NLTKstopWords])
    df['text'] = df['text'].apply(removeStops)
    df['text'] = df['text'].str.lower()

    # do a minimal version of clean up for title
    df['title'] = df['title'].apply(lambda x: re.sub('(?<=\D)[.,]|[.,](?=\D)', ' ', x))
    df['title'] = df['title'].apply(removeStops)
    df['title'] = df['title'].str.lower()
    
    return(df)
    
def assembleOneDataFramePerSymbol(symbol):
    f1name = symbol + '_reuters_news.csv'
    f2name = symbol + '_seekinga_news.csv'
    path = r'C:\Users\popaz\Documents\DataScience\ScrapeProject\crispr'
    
    df_reuters = loadCleanCsv(os.path.join(os.sep,path,f1name))
    df_seeking = loadCleanCsv(os.path.join(os.sep,path,f2name))
    df = pd.concat([df_reuters, df_seeking], ignore_index=True, axis = 0) # along rows, if we have same number cols
    
    return(df)
    
def assembleOneDataFrame():
    df_edit = assembleOneDataFramePerSymbol('edit')
    df_ntla = assembleOneDataFramePerSymbol('ntla')
    df_crsp = assembleOneDataFramePerSymbol('crsp')
    df = pd.concat([df_edit, df_ntla, df_crsp], ignore_index=True, axis=0) 
    df = collapseDateDuplicates(df) # concatenate news on same day from same symbol, source
    return(df)

def symbol2companyname(symbol):
    toName = {'edit':'editas', 'crsp':'crispr', 'ntla':'intellia' }
    return(toName[symbol])
    
def bagOfWordsPrep(df):
    # numberical values do not matter for topic discovery, 
    # but their mere presence may. So replace them by constant string 'nnn'. 
    numRegexp = re.compile(r' [1-9]\d*(\.\d+)?') 
    df['text_anum'] = df['text'].apply(lambda x: numRegexp.sub(' nnn', x))
    # 'cas', 'cas-9' and 'cas 9' are the same (refer to scissor in crispr-cas9)
    df['text_anum'] = df['text_anum'].apply(lambda x: re.sub(r'cas[ \-\d|\d\*i]+', 'cas ', x))
    # if we add the title, we may reinforce the important words and improve topic modeling
    df['text_anum'] = df['text_anum'] + df['title'].apply(lambda x: numRegexp.sub(' nnn', x))
    return(df)
    
def featureExtractFromTextCol(colTexts):
    # return a sparse matrix: one row for each document, one column for each word
    # the element (i,j) counts how many times word j appeared in document i. 
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(colTexts)
    print(cv_fit.shape)
    return(cv_fit, cv.get_feature_names()) # 254 by 2552 when df is 254 by 6. Makes sense!
    
def ldaTopicModel(cv_fit):
    # return the model
    no_topics = 3
    lda = LatentDirichletAllocation(        \
            n_components = no_topics,       \
            max_iter = 30,                  \
            learning_method = 'online',     \
            learning_offset = 50.0,         \
            random_state = 0).fit(cv_fit)
    return(lda)

def displayTopics(model, features, no_top_words):
    # Adapted from: 
    # https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
    for topicIdx, topic in enumerate(model.components_):
        print("Topic %d:"%(topicIdx))
        sortedIdx = topic.argsort()
        nPerTopic = len(topic)
        print(list([features[i] for i in sortedIdx[nPerTopic-1:nPerTopic-10:-1]]))
        
def posnegListsFromDictionary():
    # The Loughran McDonald dictionary of financial terms can be downloaded
    # at https://sraf.nd.edu/textual-analysis/code/. 
    # Derive lists of positive and negative words from the dictionary. 
    # Note the dictionary also has uncertain, litigious etc categories, but 
    # I ignore those categories in the first cut. 
    LM = pd.read_csv(r'C:\Users\popaz\Documents\DataScience\ScrapeProject\crispr\LoughranMcDonald_MasterDictionary_2014.csv')
    posSet = set()
    negSet = set()
    for sentiSign in ['Positive', 'Negative']:
        curSet = posSet if sentiSign=='Positive' else negSet 
        terms = LM['Word'][LM[sentiSign]>0]
        for t in terms: 
            curSet.add(t)
    return posSet, negSet

def scoreText(text, plusSet, minusSet):
    # returns score for the text, to be applied once per doc
    
    # use lemmatized version of word if word is not found
    lemztr = WordNetLemmatizer()
    score = 0
    twords = re.split(r'\s', text)
    for t in twords:
        t = t.upper()
        if (t in plusSet) or (lemztr.lemmatize(t) in plusSet):    
            score += 1
        if (t in minusSet) or (lemztr.lemmatize(t) in minusSet):
            score -= 1
            
    return(score)

def loadReturns(symbol):
    # returns data frames of stock returns, two columns: Date, Return
    # returns available until 2018-04-09.
    path = r'C:\Users\popaz\Documents\DataScience\ScrapeProject\Latest\stocks\stocks' 
    fname = symbol + '.us.txt'
    
    df = pd.read_csv(os.path.join(os.sep, path, fname))
    # Crucial: the Date field should be datetime not object (string).
    # This will facilitate subsequent joins and queries. 
    df = consistentDateTime(df, 'Date')
    
    df = df[['Date', 'Close']] # return series is based on close prices
    df[['Close']] = df[['Close']].pct_change()
    df.dropna(subset=['Close'], inplace=True)
    df.rename(columns={'Date':'date','Close':'return'}, inplace=True)
    
    # Note the stocks analyzed here have no dividends, no corporate actions. 
    # Otherwise, the return series would have to be div, ca-adjusted. 
    return(df)
    


def hitRatio(symbol, allnewsDF, lagDays, retSpanDays):
    # computes the proportion of same-direction coincidences 
    # between positive news and subsequent returns
    companyName = symbol2companyname(symbol.lower())
    retsDF = loadReturns(symbol)
    raw_newsDF = allnewsDF[allnewsDF['searchtext']==companyName]
    
    if(retSpanDays>1):
        print('not supported yet')
    
    # Keep only the necessary columns (and 'title' for debugging purpose)  
    newsDF = raw_newsDF[['date', 'source', 'title', 'score']]
    # There could be news from different sources on the same date. Group and keep average score. 
    grouped = newsDF.groupby('date')
    newsDF = grouped.agg({'score':np.mean}).reset_index() # only keep date, score
        
    # index by date to prepare for lags, join
    retsDF.sort_values(by=['date'], inplace=True, ascending = True)
    newsDF.sort_values(by=['date'], inplace=True, ascending = True)
    retsDF.set_index('date', inplace=True)
    newsDF.set_index('date', inplace=True)
    
    # Forward fill the signal (news score). 
    news_start = newsDF.index.min()
    news_end = newsDF.index.max()
    dates = pd.date_range(news_start, news_end, freq='D')
    dates.name = 'date'
    newsDF = newsDF.reindex(dates, method='ffill')
    
    # align signal (news) with realized returns on day T. At lag=1 day, the overlap is
    # not perfect, unless we are ready to buy into close of T-1 and sell into close of T.
    lagged_newsDF = newsDF.shift(periods=lagDays)
    combinedDF = pd.merge(lagged_newsDF, retsDF, left_index=True, right_index=True)
    combinedDF['hit'] = combinedDF['score'].apply(np.sign) * combinedDF['return'].apply(np.sign)
    num_hits    = combinedDF[combinedDF['hit'] ==  1].shape[0] # number of rows
    num_misses  = combinedDF[combinedDF['hit'] == -1].shape[0] # number of rows
    hitRatio    = num_hits/(num_hits + num_misses) # need to compute as proportion
    return(hitRatio)
    
### topic inference. 
df = assembleOneDataFrame()
df = bagOfWordsPrep(df)
sparseMatrix, features = featureExtractFromTextCol(df['text_anum'])
model = ldaTopicModel(sparseMatrix)
displayTopics(model, features, 10)

# bag-of-words sentiment, based on Loughran McDonald
plusSet, minusSet = posnegListsFromDictionary()
df['score'] = df['text_anum'].apply(lambda x: scoreText(x, plusSet, minusSet))

# Compute hit ratio with returns lagged. 
# But that means I would break the 254 rows into 3 ticker-specific sets. 
# Comparison could be made against lag=1 span=1day return, or lag=1, span=7day return
editHR = hitRatio('edit', df, 1, 1)
ntlaHR = hitRatio('ntla', df, 1, 1)
crspHR = hitRatio('crsp', df, 1, 1)
print("Hit Ratios: editas=%.2f, ntla=%.2f, crsp=%.2f" % (editHR, ntlaHR, crspHR))

# Can we do better than the hit ratios above ?


