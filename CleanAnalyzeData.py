# -*- coding: utf-8 -*-
"""
Created on Sun May 13 22:36:22 2018

@author: popaz
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import WordNetLemmatizer
import matplotlib
from MariusTagging import MariusTagger


# This file contains code that I used to predict the returns of CRISPR-related 
# companies based on natural language processing of relevant news. 
# This approach already shows profit potential, but it fails at negatives: 
# news that a paper that had thrown doubt on the CRISPR approach was withdrawn 
# is not interpreted as a positive. I need to add POS tagging,
# or use word2vec embedding from deep learning, to overcome this challenge. 

def consistentDateTime(df, timeColName):
    #Utility to apply the same standard treatment to datetimes, e.g.
    #force-cast the column from obj type into datetime stamp. 
    #This column will be used as a join key, it's more robust to have a 
    #a numeric column from joins than a string one. 
    
    # some rows are empty. if the date is empty, the row is empty. 
    df = df[pd.notnull(df[timeColName])]
    df[timeColName] = pd.to_datetime(df[timeColName])    
    df[timeColName] = df[timeColName].dt.normalize() # removes the timestamp
    return(df)
    
def collapseDateDuplicates(df):
    # only applies to the 'news' data frame. the returns won't have duplicates. 
    tmpgroup = df.groupby(["date", "searchtext", "source"])
    df = tmpgroup.agg({'text':' '.join, 'title': ' '.join, 'title_raw': ' . '.join}).reset_index() #concatenates strings
    return(df)
    
def textColClean(df, colName, dropPunctuation=False, dropStopwords=True, dropTickers=True):
    # Standard cleaning for a text column: standardize separators etc. 
    # Punctuation removal is an option; we may want to leave punctuation
    # if we follow a sentence-parsing approach as opposed to a pure bag-of-words
    # approach. 
    punctRegexp = re.compile(r'[\s:-]+') 
    df[colName] = df[colName].str.lower()
    # replace some separators with spaces
    df[colName] = df[colName].apply(lambda x: punctRegexp.sub(' ', x))
    # drop any non-ascii encodings (e.g. unicode characters)
    df[colName] = df[colName].apply(lambda x: x.encode('ascii', errors='ignore').decode())
    # 22.75PER SHARE should be 22.75 PER SHARE
    
    if dropPunctuation: 
        # eliminate dot as punctuation, not inside decimals
        df[colName] = df[colName].apply(lambda x: re.sub('(?<=\D)[.,]|[.,](?=\D)', ' ', x))
    
    # normalize the crispr/cas occurence
    df[colName] = df[colName].apply(lambda x: re.sub('crispr[\-/]cas', 'crispr cas', x))
    
    # normalize full company names
    df[colName] = df[colName].apply(lambda x: re.sub('editas medicine', 'editas', x))
    df[colName] = df[colName].apply(lambda x: re.sub('intellia therpeutics', 'intellia', x))
    df[colName] = df[colName].apply(lambda x: re.sub('crispr therpeutics', 'crispr', x))
    df[colName] = df[colName].apply(lambda x: re.sub('editing', 'edit', x))
   
    if dropStopwords: 
        # remove stop words
        NLTKstopWords = set(stopwords.words('english'))
        # add my own stop words to the list
        stopWords = NLTKstopWords | {'requires', 'related'}
        # add tickers and detritus to the removal list
        stopWords = stopWords | {'cara', 'edit', 'ntla', 'crsp', 'cytk', 'ecyt', 'kura', 'neos', 'ld'}
        removeStops = lambda x: ' '.join([word for word in x.split() if word.lower() not in stopWords])
        df[colName] = df[colName].apply(removeStops)
    
    # fix punctuation attached to words e.g. "(leading)" -> "( leading )" 
    df[colName] = df[colName].apply(lambda x: re.sub('(["\.,\(]+)([a-z]+)', r'\1 \2', x))
    df[colName] = df[colName].apply(lambda x: re.sub('([a-z]+)(["\.,\)]+)', r'\1 \2', x))

    return(df)


def loadCleanCsv(csvName):
    df = pd.read_csv(csvName)
    # Crucial: the Date field should be datetime not object (string)
    # This will facilitate subsequent joins and queries. 
    df = consistentDateTime(df, 'date')
    # Create a copy of the title where we keep punctuation and stop words. 
    df['title_raw'] = df['title']
    df = textColClean(df, 'text', dropPunctuation=True)
    df = textColClean(df, 'title', dropPunctuation=True)
    df = textColClean(df, 'title_raw', dropPunctuation=False, dropStopwords=False)
          
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
    
def tagTitles(df, rawColName, tagColName):
    # apply tagger to the data in rawColName
    # create a new column containing the tagged data. 
    tagger = MariusTagger()
    df[tagColName] = df[rawColName].apply(lambda x: tagger.tagText(x))
    return(df)

def scoreTags(tagList, tagsToScore, plusSet, minusSet, lemztr, weight=1):
    # apply some simple scoring to tagged text.
    # only score the tags in tagsToScore
    splus = sum([+1 for (word,tag) in tagList if tag.upper() in tagsToScore and (word.upper() in plusSet or lemztr.lemmatize(word.upper()) in plusSet)])
    sminus = sum([-1 for (word,tag) in tagList if tag.upper() in tagsToScore and (word.upper() in minusSet or lemztr.lemmatize(word.upper()) in minusSet)])
    return(weight*(splus+sminus))

def scoreTitles(df, tagColName, tagScoreName):
    # create a new tag column with scores based on column with tagged text
    plusSet, minusSet = posnegListsFromDictionary()
    lemztr = WordNetLemmatizer()
    df[tagScoreName] = df[tagColName].apply( \
      lambda x: scoreTags(x, ['NN','VB','VBN','VBZ','VBG'], plusSet, minusSet, lemztr, weight=10))
    return(df)
       
def bagOfWordsPrep(df):
    # numberical values do not matter for topic discovery, 
    # but their mere presence may. So replace them by constant string 'nnn'. 
    numRegexp = re.compile(r'[ \$+\-(][0-9]\d*([\.,]\d+)?') 
    df['text_anum'] = df['text'].apply(lambda x: numRegexp.sub(' nnn ', x))
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
    # TODO: centralize the numeric constants up top
    numTopics = 3
    lda = LatentDirichletAllocation(        \
            n_components = numTopics,       \
            max_iter = 40,                  \
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

# Generate lists of positive and negative words        
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
            
    #Add my own words that are more applicable to finance        
    posSet.update(['RALLY', 'INCREASE', 'SURGE', 'BEATS'])   
    negSet.update(['PLUMMET', 'FALL', 'MISSES'])     
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
    
def alignSignalReturns(symbol, allnewsDF, lagDays, retSpanDays):
    # Align the signal with the return stream. 
    # Return the combined signal-return data frame for further processing
    companyName = symbol2companyname(symbol.lower())
    retsDF = loadReturns(symbol)
    raw_newsDF = allnewsDF[allnewsDF['searchtext']==companyName]
    
    if(retSpanDays>1):
        print('not supported yet')
    
    # Keep only the necessary columns (and 'title' for debugging purpose)  
    newsDF = raw_newsDF[['date', 'source', 'title', 'score', 'title_scored']]
    # There could be news from different sources on the same date. Group and keep average score. 
    grouped = newsDF.groupby('date')
    newsDF = grouped.agg({'score':np.mean, 'title_scored':np.mean}).reset_index() # only keep date, score
        
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
    return(combinedDF)

def plotPerformance(combinedDF, symbol, scoreCol = 'score'): 
    # Plots the performance of the strategy versus the base line (long stock). 
    # Also returns Sharpe Ratio. 
    combinedDF['stra_rets'] = combinedDF[scoreCol].apply(np.sign)*combinedDF['return']
    straLevels = np.cumprod(combinedDF['stra_rets'] + 1)
    longLevels = np.cumprod(combinedDF['return'] + 1)
    
    startDate = pd.to_datetime('2016-02-03') # hard-wired for now
    startPoint = pd.Series({startDate:1})
    straLevels = startPoint.append(straLevels)
    longLevels = startPoint.append(longLevels)
    
    matplotlib.pyplot.plot(straLevels.index, straLevels.values, label = symbol + '.Strategy')
    matplotlib.pyplot.plot(longLevels.index, longLevels.values, label = symbol + '.Long-only')
    matplotlib.pyplot.legend(loc='upper center')
    matplotlib.pyplot.title('Levels:' + symbol)
    matplotlib.pyplot.show()
    
    sharpe = np.sqrt(252)* np.mean(combinedDF['stra_rets']) / np.std(combinedDF['stra_rets'])
    return(sharpe)

def hitRatio(combinedDF, scoreCol='score'):
    # computes the proportion of same-direction coincidences 
    # between positive news and subsequent returns    
    combinedDF['hit'] = combinedDF[scoreCol].apply(np.sign) * combinedDF['return'].apply(np.sign)
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

# POS-tag approach
df = tagTitles(df, 'title_raw', 'title_tagged')
df = scoreTitles(df, 'title_tagged', 'title_scored')
# f['score'] = df['score'] + df['title_scored']
# Compute hit ratio with returns lagged. 
# But that means I would break the 254 rows into 3 ticker-specific sets. 
# Comparison could be made against lag=1 span=1day return, or lag=1, span=7day return
eComb   = alignSignalReturns('edit', df, 1, 1)
editHR  = hitRatio(eComb)
editHRt = hitRatio(eComb, scoreCol = 'title_scored')

nComb   = alignSignalReturns('ntla', df, 1, 1)
ntlaHR  = hitRatio(nComb)
ntlaHRt = hitRatio(eComb, scoreCol = 'title_scored')

cComb   = alignSignalReturns('crsp', df, 1, 1)
crspHR  = hitRatio(cComb)
crspHRt = hitRatio(cComb, scoreCol = 'title_scored')
print("Hit Ratios: editas=%.2f, ntla=%.2f, crsp=%.2f" % (editHR, ntlaHR, crspHR))
print("Hit Ratios (title only): editas=%.2f, ntla=%.2f, crsp=%.2f" % (editHRt, ntlaHRt, crspHRt))

# Can we do better than the hit ratios above ?
editSR      = plotPerformance(eComb, 'editas')
editSR_T    = plotPerformance(eComb, 'editas', scoreCol = 'title_scored')
ntlaSR      = plotPerformance(nComb, 'ntla', scoreCol = 'title_scored')
crspSR      = plotPerformance(cComb, 'crsp', scoreCol = 'title_scored')
print("Sharpe Ratios: editas=%.2f, ntla=%.2f, crsp=%.2f" % (editSR, ntlaSR, crspSR))
print("Sharpe Ratios (title only): editas=%.2f" % (editSR_T))
