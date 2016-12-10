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
from mongo import *
import html.parser as hp

class Spider:
    pageList = ['/1229327625\\/EfeTCqxgo?ref=page']
    weiboList = [{'id':'EfeTCqxgo','uid':'1229327625'}]
    commentPagetList = ['http://weibo.com/aj/v6/comment/big?ajwvr=6&id=4036321846324396']
    commentList = []
    userList = []
    WBLogin = weiboLogin()

    username = 'hatikoy@163.com'
    password = 'maomao0000weibo'

    # 打开热门微博
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
                content = p3.replace('nbsp','')  # 替换实体
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

                # 保存到List
                self.commentPagetList.append(commentsURL)

            except:
                print("Failed to get weibo")

    def parseComment(self):
        for i in range(len(self.commentPagetList)):
            response = urllib2.urlopen(self.commentPagetList[i])
            toID = self.weiboList[i]['uid']
            weiboID = toID
            text = response.read().decode('utf-8', 'ignore');
            data = json.loads(text)
            comments = data['data']['html']
            #print(comments)
            pageCount = data['data']['page']['totalpage']
            comments.replace('\n','')
            print('Parse Comments:')
            print(i)
            # 用户id
            p1 = re.findall('usercard\S*',comments)
            for item in p1:
                # 如果不是一个回复
                if item.find('id')!= -1:
                    userID = item[item.rfind('=') + 1:item.rfind('"')]
                    # 削减
                    comments = comments[comments.find('ucardconf')+10:]
                    # 获取内容
                    content = comments[comments.find('</a>')+4:comments.find('</div>')]
                    # print("content:")
                    # print(content)
                    # 获取时间
                    pos = comments.find('WB_from')
                    t1 = comments[pos:]
                    time = t1[t1.find('>')+1:t1.find('</div>')]
                    # 获取昵称
                    n1 = comments[comments.find('>')+1:comments.find('</a>')]
                    comment = {
                        'id':userID,
                        'toID':toID,
                        'weiboID':weiboID,
                        'time':time,
                        'content':content
                    }
                    user = {
                        'id':userID,
                        'nick':n1,
                        "url":"http://www.weibo.com/"+userID
                    }
                    self.commentList.append(comment)
                    self.userList.append(user)

                # 如果是一个回复 这里没有处理好，就不做了
            # 加载完一页回复，保存


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

        # print(self.weiboList)
        self.parseComment()
        saveCom(self.commentList)
        saveUser(self.userList)
        # saveCom(self.commentList)
        # saveUser(self.userList)



