import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
import matplotlib.ticker as mticker
from pytz import timezone
import time
import datetime
from StringIO import StringIO
import re
from multiprocessing import Process, freeze_support


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def parseDate(inputDate):
    dateComponents = [int(token.strip()) for token in re.split("\(|\)|,",inputDate.strip()) if token.strip().isdigit()]
    parsedDate = datetime.datetime(*dateComponents[:-1])
    return parsedDate

def extractData(inputData):
    # 
    for line in inputData:
        if line.strip() and not line.startswith("%"):
            datimeRegex = re.compile(r"datime\(.*?\)")
            dates = datimeRegex.findall(line)
            startDate = parseDate(dates[0])
            endDate = parseDate(dates[1])
            tokens = [rawToken.strip() for rawToken in re.split('[(),\[\]]',line.strip()) if rawToken.strip()]
            if tokens:
                extracted = tokens[1:4]
                extracted.extend(tokens[5:7])
                extracted.append(str(time.mktime((startDate + datetime.timedelta(hours=3)).timetuple())))
                extracted.append(str(time.mktime((endDate + datetime.timedelta(hours=3)).timetuple())))
                yield ','.join(extracted)+"\n"

def timelines(y, xstart, xstop, color='b', min_timestamp=0):
    plt.scatter(xstart-min_timestamp,y,c=color,marker=".",lw=2,edgecolor=color)
    
def plotEventStream(inputFilepath, isInput):
    # read data
    inputStream = open(inputFilepath)
    data = np.genfromtxt(extractData(inputStream), 
        names=['input_type', 'user', 'input_value', 'last_update', 'confidence', 'start_time', 'end_time'], dtype=None, delimiter=',')
    input_type, user, input_value, last_update, confidence, start_time, end_time = data['input_type'], data['user'], data['input_value'], data['last_update'], data['confidence'], data['start_time'], data['end_time']

    # input and output files are dispayed differently
    if not isInput:
        # collect all instances
        input_combos = data[['input_type', 'user', 'input_value', 'last_update', 'confidence', 'start_time', 'end_time']]

        # get positions of unique input_types
        user_input_types, indices = np.unique(input_combos[['input_type', 'user', 'input_value', 'start_time']][input_combos['input_type'] == 'hla'], return_inverse = True)

        # create reverse dict from unique input type index to indices in
        # original array where the unique input type appears
        index_groups = {}
        for i in range(len(indices)):
            if indices[i] in index_groups:
                index_groups[indices[i]].append(i)
            else:
                index_groups[indices[i]] = [i]

        unique_type_instances = {}
        min_start_time = None

        for unique_idx in index_groups:
            # for each unique input type
            type_instances = input_combos[index_groups[unique_idx]]

            # sort instances by interval length
            sorted_type_instances = sorted(type_instances, key = lambda input: float(input['end_time']) - float(input['start_time']), reverse=True)

            # get the max one
            max_type_instance = sorted_type_instances[0]
            if not min_start_time:
                min_start_time = float(max_type_instance['start_time'])
            elif float(max_type_instance['start_time']) < min_start_time:
                min_start_time = float(max_type_instance['start_time'])

            # create an input type key (which excludes the start times,
            # i.e. all separate instances for the same input_type, user, 
            # input_value combination will be included in this list
            key = max_type_instance['input_type'] + "(" + max_type_instance['user'] + ", " + max_type_instance['input_value'] + ")"
            if key in unique_type_instances:
                unique_type_instances[key].append(max_type_instance)
            else:
                unique_type_instances[key] = [max_type_instance]

        # draw the plot
        ax = plt.gca()
        plt.gcf().canvas.set_window_title("Output data")

        xticks = []
        yticks = []
        y = 1
        for key in unique_type_instances:
            # yticks are the same as the input type keys
            yticks.append(key)
            unique_type_instances[key] = sorted(unique_type_instances[key], key = lambda x: float(x['start_time']))
            color_idx = 0
            for instance in unique_type_instances[key]:
                # xticks are start_time and end_time timestamps
                xticks.append(float(instance['start_time']) - min_start_time)
                xticks.append(float(instance['end_time']) - min_start_time)
                plt.hlines(y, float(instance['start_time']) - min_start_time, float(instance['end_time']) - min_start_time, COLORS[(color_idx - 1) % len(COLORS)], lw = 3)
                color_idx += 1
            y += 1

        xticks = sorted(xticks)

        plt.xticks(xticks, rotation=90)
        plt.ylim(0, y)
        plt.yticks(range(1, y + 1), yticks)
        plt.xlabel('Time')

        delta = (xticks[-1] - xticks[0]) / 20
        plt.xlim(xticks[0] - delta, xticks[-1] + delta)

        plt.title("High Level Activities")
        plt.show()

    else:
        # collect all instances
        input_combos = data[['input_type','input_value']]

        # get positions of unique input_types
        input_types, unique_idx, input_type_markers = np.unique(input_combos, 1, 1)
        y = (input_type_markers + 1) / float(len(input_types) + 1)

        # Plot data
        color_idx = 0
        min_timestamp = min(start_time)
        for input_ in input_types:
            typeFilter = (data[['input_type','input_value']] == input_)
            timelines(y[typeFilter], start_time[typeFilter], end_time[typeFilter], COLORS[color_idx], min_timestamp)
            color_idx += 1
            color_idx %= len(COLORS)

        # Setup the plot
        ax = plt.gca()
        plt.gcf().canvas.set_window_title("Input data")

        #To adjust the xlimits a timedelta is needed.
        delta = (end_time.max() - start_time.min())/20

        xticks = np.arange(0, end_time.max()-start_time.min(), 10.0)
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(xticks))
        
        input_names = []
        for input_ in input_types:
            input_names.append(input_[0].upper() + "(" + input_[1] + ")")
        plt.yticks(y[unique_idx], input_names)
        plt.ylim(0,1)
        plt.xlim(0-delta, end_time.max()-start_time.min()+delta)
        plt.xlabel('Time')
        plt.title("Position & Low Level Activities")
        plt.show()

if __name__ == '__main__':
    # run the plots in parallel
    freeze_support()
    f = "../input.stream"
    g = "../hla_output.stream"
    p1 = Process(target=plotEventStream,args=([f, True]))
    p2 = Process(target=plotEventStream,args=([g, False]))
    p1.start()
    p2.start()
    p1.join()
    p2.join()