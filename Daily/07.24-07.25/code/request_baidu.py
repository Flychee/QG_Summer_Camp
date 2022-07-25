#爬网页标题
import re
import requests
from lxml import etree
import time
for i in range(10):
    time.sleep(1)
    url = "https://www.baidu.com/s?wd=qg%E5%B7%A5%E4%BD%9C%E5%AE%A4&pn={}".format(i * 10)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    text = response.text
    with open("./baidu{}.html".format(i), "w+", encoding="utf8") as fp:
        fp.write(text)
    print("写入成功")