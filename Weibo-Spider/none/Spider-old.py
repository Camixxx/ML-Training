# -*- coding: utf-8 -*-
########################
## author:camixxx
## date:2016/10/23
## weiboLogin.py
########################

import sys
from weiboLogin import *
import urllib.request as urllib2
import urllib
import urllib.parse
import  re
import json
import lxml.etree as etree
import html
from bs4 import BeautifulSoup
import html.parser as hp
from mongo import *

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
    'lt':'<','60':'<',
    'gt':'>','62':'>',
    'amp':'&','38':'&',
    'quot':'"','34':'"',}

    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
            entity=sz.group()#entity全称，如&gt;
            key=sz.group('name')#去除&;后entity,如&gt;为gt
    try:
                htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
                sz=re_charEntity.search(htmlstr)
    except KeyError:
                 #以空串代替
                htmlstr=re_charEntity.sub('',htmlstr,1)
                sz=re_charEntity.search(htmlstr)
    return htmlstr

class myHTMLParser(hp.HTMLParser):
    a_t=False
    def handle_starttag(self, tag, attrs):
        if str(tag).startswith("div"):
            print("开始一个标签:", tag)
            for attr in attrs:
                print(attr)
                if attr == ('node-type','feed_list_content'):
                    self.a_t = True
                    print("OK")


    def handle_endtag(self, tag):
        if tag == "div":
            self.a_t=False
            print("结束一个标签:",tag)

    def handle_data(self, data):
        if self.a_t is True:
            print("得到的数据: ",data)



# 解析
def parseIndex(text):
    print("Parsing...")
    fp_raw = open("weibo.html",'r',encoding='utf-8')
    text = fp_raw.read()
    fp_raw.close()



def parseComment(url):
    # 获取并过滤出评论内容，存在pagecont数组
    # pagecont = {}
    # pagecontent = re.findall(r'<p class=\\\"detail\\(.*?)<\\\/p>', page)
    # for t in range(0, len(pagecontent)):
    #     a = pagecontent[t].split("<\/a>")
    #     b = a[len(a) - 1]
    #     c = re.sub(r"<img(.*?)>", '[表情]', b)  # 去掉图片表情
    #     d = re.sub(r"<span(.*?)span>", '', c)
    #     pagecont[t] = re.sub(r"\\t|:|：", '', d)  # 去掉最后的/t和最前的冒号

    ''' aj /v6/comment/conversation?ajwvr=6&cid=4035217071643913&type = small & ouid = & cuid = & is_more = 1 & __rnd = 1477554989888
    ".
    http: // weibo.com / 1886419032 / EeK4JA7uK
    '''
    request = urllib2.Request(url,None,weiboLogin.headers)
    response = urllib2.urlopen(request)
    text = response.read().decode('UTF-8', 'ignore');
    print(text)



def main():

    WBLogin = weiboLogin()
    print('Login:')
    username = 'hatikoy@163.com'
    password = 'maomao0000weibo'

    # 登陆 获取重定向
    login_url = WBLogin.start(username, password)
    print("After redercting:");
    print(login_url)
    try:
        req = urllib2.Request(login_url)
        test = urllib2.urlopen(req).read()
        print(test)
        print("Login success!")
    except:
        print('Login error!')

    #加载页面
    url =' http://weibo.com/2656274875/Ef3Fnapc4?ref=page_102803_ctg1_1760_-_ctg1_1760_home&rid= 0_0_0_2676202755072789896'
    req = urllib2.Request(url)
    page = urllib2.urlopen(req).read().decode('utf-8','ignore')
    match = re.search("ouid=[0-9]*&location=&comment_type=0",page)
    commentsURL= 'http://weibo.com/aj/v6/comment/big?ajwvr=6&'+page[match.start()+2:match.end()-25]
    print(commentsURL)
    print("---------------------------------")

    #request = urllib2.Request(commentsURL)
    response = urllib2.urlopen(commentsURL)
    text = response.read().decode('utf-8','ignore');
    data = json.loads(text)
    comments = data['data']['html']

    print("---------------------------------")

    # 获取某微博页面
    # pageURL = 'http://weibo.com/2642332004/EeMMr3GyK' +'?type=comment#'
    # pageContent = WBLogin.callUrl(pageURL)
    # print("page ***************")
    # print(pageContent)
    # print("page over **********")

    # #解析对话


     # conversationURL= 'http://weibo.com/aj/v6/comment/conversation?ajwvr=6&cid='+'4035266081744815'+'&type=small&ouid=&cuid=&is_more=1&__rnd=1477595592346'

    # # 解析评论
    # p = re.compile('value=comment:[0-9]*')
    # comid = p.search(pageContent)
    # id = pageContent[comid.start()+14:comid.end()]
    #
    # # 获取评论
    # commentURL = 'http://weibo.com/aj/v6/comment/big?ajwvr =6&id='+id +'& __rnd = 1477554989888'
    # commentContent = WBLogin.callUrl(commentURL)

class Spider:
    pageList = ['/5293126213\\/Ef7MFbZob?ref=page']
    weiboList = []
    commentPagetList = []
    commentList = []
    userList = []
    WBLogin = weiboLogin()

    username = 'hatikoy@163.com'
    password = 'maomao0000weibo'

    def openHot(self):
        print("Open Hot Page")
        url = ' http://weibo.com/feed/hot'
        req = urllib2.Request(url)
        res = urllib2.urlopen(req).read().decode('UTF-8')
        html = res[res.find('feed内容'):res.rfind('feed内容')]
        pageList = re.findall('[0-9]*/\S*ref=page', res)

        # for i in range(1,len(pageList)) :
        #     pageList[i].replace('\\','')
        print(pageList)
        self.pageList.extend(pageList)
        return pageList

    def getWeibo(self):
        for i in self.pageList:
           # i.replace('\\', '')

            # 用户号
            try:
                text = re.findall('[\w]*', i)
                userID = text[1]
                weiboID = text[4]
                pageURL = "http://weibo.com"+'/'+userID+'/'+weiboID
                print("userID:"+userID+'|weiboID:'+weiboID)
                print(pageURL)

                # 打开页面
                page = urllib2.urlopen(pageURL).read().decode('utf-8','ignore')

                # 解析页面
                print("Parsing...")
                page = page.replace('\\','')
                page = page.replace('\n', '')

                # 微博内容获取 昵称获取
                p1 = page[page.find('feed_list_content'):]
                p2 = p1[p1.index('=')+1:p1.index('">')]

                p3 = p1[p1.index('>')+2:p1.index('</div>')]
                p4 = re.compile('</?\w+[^>]*>')  # 去掉标签
                p3 = p4.sub('',p3)
                content = replaceCharEntity(p3)  # 替换实体
                nickName = p2

               # 时间获取
                p5 = page[page.find('time"')+5:page.find('" date')]
                time = p5[p5.find('"'):]

                weibo = {
                    'id':weiboID,
                    'uid':userID,
                    'content':content,
                    'time':time,
                    "url":pageURL
                }
                user = {
                    'id':userID,
                    'nick':nickName,
                    "url":"http://www.weibo.com/"+userID
                }
                # 保存到List
                self.weiboList.append(weibo)
                self.userList.append(user)

                # 获取评论
                match = re.search("ouid=[0-9]*&location=&comment_type=0", page)
                commentsURL = 'http://weibo.com/aj/v6/comment/big?ajwvr=6&' + page[match.start() + 2:match.end() - 25]
                print("commentsURL:");print(commentsURL)
                # request = urllib2.Request(commentsURL)

                self.commentPagetList.append(commentsURL)

                # response = urllib2.urlopen(commentsURL)
                # text = response.read().decode('utf-8', 'ignore');
                # data = json.loads(text)
                # comments = data['data']['html']
                # print(comments)
                # print("---------------------------------")

            except:
                print("Failed to get user id")
    def start(self):
        print('Login:')
        # 登陆 获取重定向
        login_url = self.WBLogin.start(self.username, self.password)
        print("After redercting:");
        print(login_url)
        try:
            req = urllib2.Request(login_url)
            test = urllib2.urlopen(req).read()
            print(test)
        except:
            print('Error')

        self.openHot()
        self.getWeibo()
        saveWeibo(self.weiboList)
        print(self.pageList)
        print("_______________")


sp=Spider()
sp.start()

