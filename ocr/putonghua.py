#!/usr/bin/env python
# coding:utf-8


from urllib import request,parse
from http.cookiejar import CookieJar, MozillaCookieJar
from urllib.error import URLError, HTTPError
import re
import io
import os
import time
import random

import cv2

from chaojiying import Chaojiying_Client

USER_AGENT_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/73.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.50',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 OPR/66.0.3515.95',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:61.0) Gecko/20100101 Firefox/73.0',
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 OPR/66.0.3515.95',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36',
    'Mozilla/5.0 (X11; Linux i586; rv:31.0) Gecko/20100101 Firefox/73.0',
    'Mozilla/5.0 (Windows NT 6.2; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 OPR/66.0.3515.95',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Opera/9.80 (Windows NT 6.2; Win64; x64) Presto/2.12 Version/12.16',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3002.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3764.0 Safari/537.36 Edg/75.0.131.0',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.68 YaBrowser/20.3.0.1050 (beta) Yowser/2.5 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3782.0 Safari/537.36 Edg/76.0.152.0',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36 OPR/82DEA0C7D63A',
    'Mozilla/5.0 (Windows NT 10.0; rv:9001.0) Gecko/20100101 Firefox/9001.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:74.0) Gecko/20100101 Firefox/74.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:74.0) Gecko/20100101 Firefox/74.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:70.0) Gecko/20100101 Firefox/70.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:70.0) Gecko/20100101 Firefox/70.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:70.0) Gecko/20100101 Firefox/70.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/82.0.4051.0 Safari/537.36 Edg/82.0.424.0',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:19.0) Gecko/20100101 Firefox/72.0.2',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:56.0) Gecko/20100101 Firefox/79.0.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/22.0 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:74.0) Gecko/20100101 Firefox/74.0',
]

err_noinfo = r'showMsg("error", "查询不到信息")'
err_wrongcap = r'showMsg("error", "验证码不正确")'
inf_data = r'考试地点'

# cookie = CookieJar() 
# opener = request.build_opener(request.HTTPCookieProcessor)
cookie = MozillaCookieJar()
# cookie.load("cookies.txt")
opener = request.build_opener(request.HTTPCookieProcessor(cookie))

home_url = 'http://bmfw.haedu.gov.cn/jycx/pthcj_check/78'
search_url = 'http://bmfw.haedu.gov.cn/jycx/pthcj_check'

def get_captcha(opener, req_captcha):
    response = opener.open(req_captcha)
    img = io.BytesIO(response.read())

    chaojiying = Chaojiying_Client('artlover', 'suxinyanfu', '904049')  #用户中心>>软件ID 生成一个替换 96001
    # im = open('captcha.jpg', 'rb').read()                                                 #本地图片文件路径 来替换 a.jpg 有时WIN系统须要//
    res_cap = chaojiying.PostPic(img, 1902)
    if res_cap['err_str'] != 'OK':
        print("[E]验证码无法识别", res_cap)
        return get_captcha(opener, req_captcha)

    return res_cap['pic_str']

#4000 ~
start = 4233
end = 5000
while True:

    headers = {
        'User-Agent': random.choice(USER_AGENT_LIST),
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh-TW;q=0.7,zh;q=0.6',
        'Connection': 'keep-alive',
        'Host': 'bmfw.haedu.gov.cn',
        'Referer': 'http://bmfw.haedu.gov.cn/jycx/pthcj_check/78',
    }
    req_home = request.Request(home_url, headers=headers)
    req_captcha = request.Request('http://bmfw.haedu.gov.cn/code/image1?' + str(random.random()), headers=headers)
    req_search = request.Request(search_url, headers=headers)

    print("\n")
    if start >= end:
        print('[I]查询结束')
        break

    response = opener.open(req_home)
    if response.geturl() != home_url:
        print('[E]请求失败', response.geturl())
        start += 1
        continue

    code = get_captcha(opener, req_captcha)
    # code = random.randrange(10000, 99999, 1);

    time.sleep(random.randrange(1, 3, 1))

    no = "0230102" + str(start)
    dict = {"xxid":"78",
            "trueName":"谢言付",
            "zkzh": no,
            "imageId":code,
            "sfzh":""
            }
    print("尝试号码: [", no, ']', dict)

    data = bytes(parse.urlencode(dict),encoding="utf-8")

    try:
        # searching
        response = opener.open(req_search,data)
        html = response.read().decode("utf-8","ignore")
        if html.find(err_noinfo) != -1: 
            print ("[E]信息不存在", dict)
        elif html.find(err_wrongcap) != -1: 
            print ("[E]验证码错误", dict)
            continue
        elif html.find(inf_data) != -1: 
            print ("[I]查询成功，信息如下：", dict)
            f = open("result.html",'w')
            f.write(html)
            f.close()
            exit()
        else:
            print ("[E]未知错误", dict, response.geturl(), response.info())
            f = open("err-" + no + ".html",'w')
            f.write(html)
            f.close()
            continue

        start += 1

    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code, e.reason)
        print('response: ', response.info())
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
        print('response: ', response.info())
    else:
        pass
    

# response = opener.open(req_home)
# print("当前页面网址："+response.geturl())
# # exit()

# code = get_captcha(opener, req_captcha)

# dict = {"xxid":"78",
#         "trueName":"申元元",
#         "zkzh":"02301025177",
#         "imageId":code,
#         "sfzh":""
#         }

# data = bytes(parse.urlencode(dict),encoding="utf-8")

# response = opener.open(req_search,data)
# print("当前页面网址：", response.geturl())

# html = response.read().decode("utf-8","ignore")
# if html.find(err_noinfo) != -1: 
#     print ("[E]信息不存在", dict)
# elif html.find(err_wrongcap) != -1: 
#     print ("[E]验证码错误", dict)
# elif html.find(inf_data) != -1: 
#     print ("[I]查询成功，信息如下：", dict)
#     exit()
# else:
#     print ("未知错误", dict)

# with open("result.html",'wb') as output:
#     output.write(response.read())

# exit()
