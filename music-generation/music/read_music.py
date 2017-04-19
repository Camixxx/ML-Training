import pretty_midi

"""
打开xml文件后，处理音符数据，转化为数组和字典
part：声部
measure：小节，每个小节有若干个音符note
note：音符
 -pitch 音高{step，octave}可以是一个1~84的数据，0代表休止符
    octave：八度音阶
    step:
key：规定升降号，大调小调 key{fifths:升降号数目,mode:major }
clef：clef{sign:(G),line(2)}规定了一条谱子高音谱号或是低音谱号在第几条线上,如果有两个clef则说明有两条谱子
division，duration： duration/division = type的值
type：whole是全音符1，half是二分音符1/2
time：time{beats，beat-type} 以1/beat-type为一拍，一个小节有beats个拍子
"""

class midi_data():
    def __init__(self, measure_notes=None, measure_chords = None, time_signature=None):
        self.measure_notes = measure_notes
        self.measure_chords = measure_chords
        self.time_signature = time_signature

def read_data( pathname , if_measure = False , notes_in_measure = None):
    """
    :param pathname: the name of midi file
    :param if_measure: realignment the notes to measures, Default False
    :param notes_in_measure: the number of notes in first measure, Default None
    :return: a dict { measures_notes[],measures_chords[],time_signature }
    """
    midi = pretty_midi.PrettyMIDI('data/'+ pathname)

    # 默认只有一个乐器进行演出
    notes = midi.instruments[0].notes
    try:
        chords = midi.instruments[1].notes
    except:
        chords = None

    # 默认节拍不变,默认每个小节时间固定
    timeSignature = midi.time_signature_changes[0]
    # print(midi.estimate_tempo())
    if timeSignature.time == 0:
        if notes_in_measure:
            # 按照每个小节进行处理
            timeSignature.time = notes[notes_in_measure].start
        else:
            timeSignature.time = 2.0

    if if_measure:
        measures_notes = re_measure(notes,timeSignature.time)
        if chords:
            measures_chords = re_measure(chords,timeSignature.time)
        else:
            measures_chords = [chords]
    else:
        print("Didn't realign into measures!!")
        measures_notes=[notes]
        measures_chords = [chords]

    return midi_data(measures_notes,measures_chords,timeSignature)


def re_measure(notes, time):
    measures = []
    temp = []
    length = len(notes)
    temp.append(notes[0])
    multiple = 1
    for iter in range(1, length):
        if notes[iter].start >= multiple * time and  notes[iter-1].end < multiple * time :
            multiple += 1
            measures.append(temp)
            notes[iter].start -= (multiple-1) * time
            notes[iter].end -= (multiple-1) * time
            temp = [notes[iter]]
        else:
            notes[iter].start -= (multiple-1)*time
            notes[iter].end -= (multiple - 1) * time
            temp.append(notes[iter])
    print("Find %d mesasure with %d notes" % (len(measures), length))
    return measures

def regular_measure(measures):
    regular_m = []
    m = []
    for notes in measures:
        for n in notes:
          # m.extend([n.start, n.end, n.pitch/88])
            m.extend([(int)(round(n.start*50)),(int)(round(n.end*50)), n.pitch])
        regular_m.append(m)
        m = []
    return regular_m

def regular_list(measures):
    regular_m = []
    m = []
    for notes in measures:
        for n in notes:
          # m.extend([n.start, n.end, n.pitch/88])
          # m.extend([(int)(round(n.start*50))*10000+(int)(round(n.end*50))*100+ n.pitch])
          m.append((int)(round(n.start * 50)) * 10000 + (int)(round(n.end * 50)) * 100 + n.pitch)
        regular_m.append(m)
        m = []
    return regular_m

def read_measure(pathname):
    midi = read_data(pathname,True)
    return regular_measure(midi.measure_notes)

def read_list(pathname):
    midi = read_data(pathname, True)
    return regular_list(midi.measure_notes)

def __test():
    print("______601598")
    data = read_data("601598.mid",if_measure = True)
    print(data.measure_notes)
    #print("Cannon_in_D")
    #m2 = read_data("Cannon_in_D.mid")
    print("end")

