# -*- coding: utf-8 -*-
########################
## author:camixxx
## date:2016/10/23
## weiboLogin.py
########################

import sys
import urllib
import urllib.request as urllib2
import urllib.parse
import http.cookiejar as cookielib
import base64
import re
import json
import rsa
import binascii
# import requests
# from bs4 import BeautifulSoup

# 新浪微博的模拟登陆
class weiboLogin:
    # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:41.0) Gecko/20100101 Firefox/41.0'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1'}

    # 获取一个保存cookies的对象
    cj = cookielib.CookieJar()

    ## Cookies  openner
    def enableCookies(self):

        # 将一个保存cookies对象和一个HTTP的cookie的处理器绑定
        cookie_support = urllib2.HTTPCookieProcessor(self.cj)
        # 创建opener,设置handler处理http的url

        opener = urllib2.build_opener(cookie_support, urllib2.HTTPHandler)
        # 安装opener，此后调用urlopen()时会使用安装过的opener对象
        urllib2.install_opener(opener)

    ## Prelogin... Get:servertime, nonce, pubkey, rsakv
    def getServerData(self):
        url = 'http://login.sina.com.cn/sso/prelogin.php?entry=weibo&callback=sinaSSOController.preloginCallBack&su=ZW5nbGFuZHNldSU0MDE2My5jb20%3D&rsakt=mod&checkpin=1&client=ssologin.js(v1.4.18)&_=1442991685270'
        data = urllib2.urlopen(url).read().decode("utf-8")
        ## Parse...
        try:
            p = re.compile('\((.*)\)')
            json_data = p.search(data).group(1)
            data = json.loads(json_data)
            servertime = str(data['servertime'])
            nonce = data['nonce']
            pubkey = data['pubkey']
            rsakv = data['rsakv']
            return servertime, nonce, pubkey, rsakv
        except:
            print('Get severtime error!')
            return None
    # 获取加密的密码
    def getPassword(self, password, servertime, nonce, pubkey):
        rsaPublickey = int(pubkey, 16)
        key = rsa.PublicKey(rsaPublickey, 65537)  # 创建公钥
        message = str(servertime) + '\t' + str(nonce) + '\n' + str(password)  # 拼接明文js加密文件中得到
        passwd = rsa.encrypt(message.encode('utf-8'), key)  # 转换为bytes然后加密
        passwd = binascii.b2a_hex(passwd)  # 将加密信息转换为16进制。
        return passwd

    # 获取加密的用户名
    def encodeUsername(self, username):
        username_ = urllib.parse.quote(username)                # 屏蔽特殊字符
        username = base64.encodebytes(username.encode("utf-8")) # 加密
        #username = base64.encodestring(username_)[:-1]
        return username
        # 获取需要提交的表单数据

    def getFormData(self, userName, password, servertime, nonce, pubkey, rsakv):
        userName = self.encodeUsername(userName)
        psw = self.getPassword(password, servertime, nonce, pubkey)

        form_data = {
            'entry': 'weibo',
            'gateway': '1',
            'from': '',
            'savestate': '7',
            'useticket': '1',
            'pagerefer': 'http://weibo.com/p/1005052679342531/home?from=page_100505&mod=TAB&pids=plc_main',
            'vsnf': '1',
            'su': userName,
            'service': 'miniblog',
            'servertime': servertime,
            'nonce': nonce,
            'pwencode': 'rsa2',
            'rsakv': rsakv,
            'sp': psw,
            'sr': '1366*768',
            'encoding': 'UTF-8',
            'prelt': '115',
            'url': 'http://weibo.com/ajaxlogin.php?framelogin=1&callback=parent.sinaSSOController.feedBackUrlCallBack',
            'returntype': 'META'
        }
        formData = urllib.parse.urlencode(form_data)
        return formData

    # 登陆函数
    def start(self, username, psw):

        self.enableCookies()

        url = 'http://login.sina.com.cn/sso/login.php?client=ssologin.js(v1.4.18)'
        servertime, nonce, pubkey, rsakv = self.getServerData()
        formData = self.getFormData(username, psw, servertime, nonce, pubkey, rsakv)

        req = urllib2.Request(
                url=url,
                data=formData.encode("utf-8"),
                headers=self.headers
        )
        #result = urllib2.urlopen(url,data=formData.encode("utf-8"))
        result = urllib2.urlopen(req)
        text = result.read().decode('utf-8', 'ignore')
        #print("Before redercting:") ;print(text)
        # 重定位url解析
        p = re.compile('location\.replace\([\'"](.*?)[\'"]\)')
        login_url = p.search(text).group(1)

        #
        #
        # # 保存cookie
        # self.cookie_value = cj
        # # 设置保存cookie的文件，同级目录下的cookie.txt
        # filename = 'cookie.txt'
        # # 保存cookie到文件

        return login_url

        # 访问主页，把主页写入到文件中
        # url = 'http://weibo.com/u/2679342531/home?topnav=1&wvr=6'
        # url = 'http://weibo.com/feed/hot'
        # url = 'http://d.weibo.com/'
        # request = urllib2.Request(url)
        # response = urllib2.urlopen(request)
        # text = response.read().decode('UTF-8', 'ignore');
        # fp_raw = open("weibo.html",'w+',encoding='utf-8')
        # fp_raw.write(text)
        # fp_raw.close()
        #return True

    def callUrl(self,url):
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        text = response.read().decode('UTF-8', 'ignore');
        return text


