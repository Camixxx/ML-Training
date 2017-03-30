
import sys
sys.path.insert(0, "../../python/")
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
'''
打开xml文件后，处理音符数据，转化为数组和字典
part是声部，一首曲子的音轨
measure是小节，每个小节有若干个音符note
note可以是一个1~84的数据，0代表休止符
'''
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
            # print(m.attrib)

            # # 原本用于判断 几分音符
            # try:
            #     division = m.find("attributes").find("divisions").text
            #     print(division)
            # except:
            #     print("This measure dosen't have  attributes or division")

            for n in m.iter("note"):  # 音符  { type , octave , step}
                try:  # 节奏
                    t = n.find("type").text
                    type = typeName[t]
                except:
                    print("This note dosen't have type")
                    continue

                try:
                    n.find("rest")  # 休止符
                    note = {"value": 0,
                            "type": type}
                    # note = {0,type}

                except:  # 音符
                    p = n.find("pitch")

                    try:
                        alter = int(p.find("alter").text)
                    except:
                        alter = 0

                    try:  # octave 八度音阶 * step + alter
                        val = int(stepName[p.find("step").text]) * int(p.find("octave").text) + alter
                    except:
                        print("Can't find step and octave，or calculate val failed.")
                        continue

                    note = {"value": val,
                            "type": type}

                    # note = {val,type}

                measure.append(note)
            part.append(measure)

    return part



'''
随机生成音符数组和字典，保证节拍之和恰好为一个小节
'''
def generateRandom(measureNum = 12,measureTime = 1):
    part = []
    measure = []
    for i in range(measureNum):
        measure=[]
        time = 0
        while time < measureTime:
            # 随机节奏
            randPow = random.randint( 0 , 4 )
            type = 1/math.pow(2,randPow)
            if(time + type > measureTime):
                type = measureTime - time

            # 随机音阶
            value = random.randint(0,84)
            #note = {type,value}
            note={'type':type,'value':value}
            measure.append(note)
            time = time + type
        part.append(measure)

    return part


'''
转换为向量：
把一个4/4小节切分为16段的数据，每段数据都是一个16分音符
向量位数为16*8，前7位表示音符数据，最后一位表示连续性
0则为连续，1为断开
beat表示有多少个16分音符
'''
def convert2vec( part, beat = 16 ):
    vp = []
    for measure in part:
        vm = []  # vectorMeasures
        for note in measure:
            vn = bin(note['value']) # 字符串音符值 Vector Note
            vn = vn[2:]        # 去掉0b头
            vn = vn.zfill(7)   # 补零
            vn = [int(number) for number in vn] # 字符列表

            t =int( note['type'] * beat ) # type /0.0625
            for i in range(t):
                vm.extend(vn)
                vm.append(0)

            vm = vm[:len(vm)-1] # 把最后一位0变成1
            vm.append(1)

        # 检验vm中音符节奏之和是否符合条件
        if(len(vm) == beat * 8 ):
            vp.append(vm)
        else:
            print("Vector Measure 的节奏数量不对")
            print(vm,measure)
            vm=vm[:beat*8]
            vp.append(vm)

    return vp

'''
加载数据入口
'''
def loadAll(filenames):
    data = []
    label = []
    for name in filenames:
        if(name == 'random'):
            part = generateRandom(random.randint(10,13))
            label_part=[0 for i in range(len(part))]
        else:
            part = openXML(name)
            label_part = [1 for i in range(len(part))]

        vector = convert2vec(part)
        data.extend(vector)
        label.extend(label_part)

    return data,label

#
# def loadRandom(len):
#     part = generateRandom(len)
#     label_part = [0 for i in range(len(part))]
#     vector = convert2vec(part)
#     return vector,lab