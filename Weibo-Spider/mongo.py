from pymongo import MongoClient
try:
    conn = MongoClient('mongodb://localhost:27017/')
except:
    conn = MongoClient('mongodb://172.17.0.12:27017/')

db = conn.lijilan #连接数据库

def saveWeibo(List):
    for i in List:
            try:
                    db.microblog.insert({#'_id':i['id'],
                                         'from_user_oid':i['uid'],
                                         'time':i['time'],
                                         'content':i['content'],
                                         'url':i['url']
                                         })
                    print("Save Success")
            except:
                    print("ERROR!!!Save:")
                    print(i)
                    print("_____________")



def saveUser(List):
    for i in List:
        try:
            db.user.insert({#'_id': i['id'],
                             'from_user_oid': i['id'],
                             'nicknames': i['nick'],
                             'url': i['url']
                             })
        except:
            print("ERROR!!!Save:")
            print(i)
            print("_____________")

def saveCom(List):
    for i in List:
        try:
            db.comment.insert({     'from_user_oid': i['id'],
                             'to_user_oid':i['toID'],
                             'to_microblog_oid':i['weiboID'],
                             'time': i['time'],
                             'content':i['content']
                             })
        except:
            print("ERROR!!!Save:")
            print(i)
            print("_____________")