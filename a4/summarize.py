from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import pickle
import time
import matplotlib.pyplot as plt

import plotly.plotly as py

def main():
    data= open('Data.txt','rb')
    c = pickle.load(data)

    no_of_tweets = len(c)
    print('Number of Tweets collected = '+str(no_of_tweets))
    data= open('Classify_Statistics.txt','rb')
    c = pickle.load(data)
    print('Number of communities discovered:'+str(len(c)))
    print('Number of instances per class found:'+str(c))
    #print(c)

    val = []
    for k,v in c.items():
       val.append(v)
    y = [val[1]]
    z = [val[0]]
    index = 0
    x1 = [2]
    x2 = [1]
    ax = plt.subplot(111)
    ax.bar(x1, y, width=0.2, color='b', align='center')
    ax.bar(x2, z, width=0.2, color='r')
    plt.plot([], [], label='Male', color='b', linewidth=10)

    plt.plot([], [], label='Female', color='r', linewidth=10)
    #plt.hist(age, bins, histtype='bar', rwidth=0.7)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    #plt.xlabel('Users')
    #plt.xaxis('off')
    plt.ylabel('Number of Users')
    #plt.xaxis
    plt.legend()
    #plt.show()
    plt.savefig('barchart.png')
    data= open('Name.txt','rb')
    c = pickle.load(data)
    print('One example from each class:')
    for k, v in c.items():
        print(k+str(v[:1]))
    data= open('Cluster_Statistics.txt','rb')
    c = pickle.load(data)
    print('Number of Communities detected = '+str(c[0]))
    print('Number of users = '+str(c[1]))
    print('Average number of users per community:'+str(int(c[1]/c[0])))




if __name__ == '__main__':
    main()