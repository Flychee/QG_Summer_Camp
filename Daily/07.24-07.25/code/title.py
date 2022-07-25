import re
import requests
from lxml import etree

c = []
for i in range(10):
    parser = etree.HTMLParser(encoding="utf8")
    html = etree.parse(r"baidu{}.html".format(i), parser=parser)
    result = html.xpath('//div[@class="c-container"]//a')
    b = []
    for title in result:
        a = etree.tostring(title, encoding='utf8').decode('utf-8')
        a = re.sub('\n|\r\s', '', a)
        a = re.search('<em>.+?</a>', a)
        a = re.findall('match=(.*)', str(a))
        if a:
            b.append(a)
    for i in b:
        i[0] = re.sub("</em>|<em>|&#|</a>", '', str(i[0]))
        c.append(str(i[0]))
with open('qg.txt', 'w', encoding='utf8') as fp:
    for i in c:
        fp.write(i + '\n')
