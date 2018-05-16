# -*- coding: utf-8 -*-
"""
Created on Sat May 12 12:02:50 2018

@author: popaz
"""
from scrapy import Spider, Request
from crispr.items import CrisprItem
from bs4 import BeautifulSoup

class ReutersSpider(Spider):
    name = 'reuters_spider'
    allowed_urls = [ 'https://www.reuters.com/' ]
    #start_urls = [ 'https://www.reuters.com/finance/stocks/EDIT.OQ/key-developments?pn=' + str(i) for i in range(1,3) ]
    #start_urls = [ 'https://www.reuters.com/finance/stocks/NTLA/key-developments?pn=' + str(i) for i in range(1,3) ]
    start_urls = [ 'https://www.reuters.com/finance/stocks/CRSP/key-developments?pn=' + str(i) for i in range(1,4) ]

    def parse(self,response): #callback for start_urls
        newsAll = response.xpath('//div[@class="column1 gridPanel grid8"]//div[@class="feature"]')
        for newsItem in newsAll: 
            item = CrisprItem()
            item['source'] = 'reuters'
            item['date'] = newsItem.xpath('.//span[@class="timestamp"]/text()').extract_first()
            item['title'] = newsItem.xpath('.//h2/a/text()').extract_first()
            item['searchtext'] = 'crispr'
            tmpTxt = newsItem.xpath('.//edit.o/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//ntla.o/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//crsp.o/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//pvac.o/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//agn.n/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//advm.o/text()').extract_first()
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//vrtx.o/text()').extract_first() 
                
            if (tmpTxt == None or 0 == len(tmpTxt.strip())):
                tmpTxt = newsItem.xpath('.//p/text()').extract_first()    
            item['text'] = tmpTxt
            yield(item)
