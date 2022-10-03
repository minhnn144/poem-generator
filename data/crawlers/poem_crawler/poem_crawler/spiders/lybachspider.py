import scrapy

class LyBachSpider(scrapy.Spider):
    name = "lybach"
    start_urls = ['https://www.thivien.net/L%C3%BD-B%E1%BA%A1ch/Anh-V%C5%A9-ch%C3%A2u/poem-PE7aG4bpaNaOL1KBzgLqOA']

    def parse(self, response):
        for poems in response.css('div.poem-view-separated'):
            yield {
                'c_title': poems.css('h4>strong::text').get(),
                'c_content': poems.css('p.HanChinese::text').getall(),
                'title': poems.css('h4>strong::text').getall()[1],
                'content': poems.css('p')[2].css('p::text').getall(),
            }