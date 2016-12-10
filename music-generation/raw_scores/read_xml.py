
import sys
import xml.etree.ElementTree as ET
import random
import math

typeName = {
    "whole": 1.0,
    "half": 0.5,
    "quarter": 0.25,
    "eighth": 0.125,
    "sixteenth" : 0.0625,
    "32nd" : 1/32,
}
stepName = {
    "C": 1,
    "D": 3,
    "E": 5,
    "F": 6,
    "G": 8,
    "A": 10,
    "B": 11
}

def openXML(name):

    path = "raw_scores/" + name
    part = []
    measure = []
    note = {}

    # 读取xml文件
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except:
        print("Error,can't open xml at" + path)
        return False

    # 解析乐谱
    for child in root.find("part"):# 声部

        for m in  child.iter("measure"):   # 小节
            measure = []

            # # 原本用于判断 几分音符
            # try:
            #     division = m.find("attributes").find("divisions").text
            #     print(division)
            # except:
            #     print("This measure dosen't have  attributes or division")

            for n in m.iter("note"):        # 音符  { type , octave , step}
                p = n.find("pitch")
                try:
                    t = n.find("type").text
                    type = typeName[t]
                except:
                    print("This note dosen't have type")

                if n.find("rest") :  # 休止符
                    # note = {"value": 0,
                    #         "type" : type}
                    note = {0,type}
                else:                # 音符转化为数值
                    try :
                        alter = int(p.find("alter").text)
                    except:
                        alter = 0

                    val = (stepName[p.find("step").text] + alter) * int(p.find("octave").text)

                    # note = {"step": p.find("step").text,
                    #         "octave": p.find("octave").text,
                    #         "alter":  p.find("step").text,
                    #         "type": typeName[t]}

                    # note = {"value": val,
                    #         "type" : type}
                    note = {val,type}

                measure.append(note)
            part.append(measure)

    return part

def generateXML(part):
    return 0

def generateRandom(measureNum = 12,measureTime = 1):
    part = []
    measure = []
    for i in range(measureNum):
        measure=[]
        time = 0
        while time<measureTime:
            # 随机节奏
            randPow = random.randint( 0 , 5 )
            type = 1/math.pow(2,randPow)

            # 随机音
            value = random.randint(0,84)

            note = {type,value}
            measure.append(note)

            time = time + type

        part.append(measure)

    return part


