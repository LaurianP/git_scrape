# -*- coding: utf-8 -*-
"""
Created on Fri May 11 09:53:52 2018

@author: popaz
"""
from scrapy import Spider, Request
from crispr.items import CrisprItem
from bs4 import BeautifulSoup

class SeekAlphaSpider(Spider):
    name = 'seekalpha_spider'
    allowed_urls = [ 'https://seekingalpha.com/' ]
    start_urls = ['https://seekingalpha.com/symbol/NTLA?analysis_tab=focus&news_tab=news-all']

    def parse(self,response): # callback for start_urls
        newsAll = response.xpath('//div[@data-feed="news-all"]')
        #how to click button for all news ????
        newsAll = newsAll.xpath('.//li[@class="mc_list_li symbol_item"]')
        print(len(newsAll))
        print("marius"*20)
        for newsItem in newsAll: 
            item = CrisprItem()
            item['source'] = 'seekingalpha'
            item['date'] = newsItem.xpath('.//span[@class="date"]/text()').extract_first()
            item['title'] = newsItem.xpath('.//a[@class="market_current_title"]/text()').extract_first()
            item['searchtext'] = 'intellia'
            bullets = newsItem.xpath('.//span[@class="general_summary light_text bullets"]//li')
            bullets_txt = []
            for bullet in bullets: 
                bullet_txt = bullet.extract()
                bullet_soup = BeautifulSoup(bullet_txt,'html.parser')
                bullet_soup.li.unwrap()  # unwrap it from <li> </li>
                for a in bullet_soup.find_all('a'): a.unwrap() # unwrap href
                for em in bullet_soup.find_all('em'): em.unwrap() #no italics
                for f in bullet_soup.find_all('font'): f.unwrap() # no color
                for s in bullet_soup.find_all('strong'): s.unwrap() #no strong
                for d in bullet_soup.find_all('div'): d.unwrap()
                for b in bullet_soup.find_all('b'): b.unwrap()
                print(bullet_soup.contents)
                print("=suiram=" * 10)
                bullet_contents = " ".join(bullet_soup.contents)
                bullets_txt.append(bullet_contents)
                
            bullets_txt = ' '.join(bullets_txt)
            item['text'] = bullets_txt
            print(item['date'])
            yield item
                