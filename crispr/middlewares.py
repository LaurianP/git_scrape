# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
from selenium import webdriver
from scrapy.http import HtmlResponse
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
import numpy as np

class CrisprSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class CrisprDownloaderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    def __init__(self):
        self.driver = webdriver.Chrome(r'C:\Users\popaz\Documents\DataScience\Week5\Selenium\chromedriver.exe')

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def new_posts(self, driver, min_len):
        return len(driver.find_elements_by_xpath('//div[@data-feed="news-all"]//li[@class="mc_list_li symbol_item"]'))> min_len

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.
        self.driver.get(request.url)
        eps1 = np.random.uniform(0,2)
        WebDriverWait(self.driver, 1+eps1)
        self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        
        eps2 = np.random.uniform(0,2)
        WebDriverWait(self.driver, 10+eps2)
        self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        WebDriverWait(self.driver, 30 + 2*eps2)
        self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        
        news = self.driver.find_elements_by_xpath('//div[@data-feed="news-all"]//li[@class="mc_list_li symbol_item"]')
        while len(news) < 150:
            try:
                self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
                print("ScrollWhile" * 5)
                WebDriverWait(self.driver, 10).until(lambda driver: self.new_posts(driver, len(news)))
                news = self.driver.find_elements_by_xpath('//div[@data-feed="news-all"]//li[@class="mc_list_li symbol_item"]')
            except TimeoutException:
                break
        body = self.driver.page_source
        
        return HtmlResponse(self.driver.current_url, body=body, encoding='utf-8', request=request)
        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        
        
        
        
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)
