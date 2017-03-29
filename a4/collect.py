from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import pickle
import time


consumer_key = 'b5xU4Rk15ivIXa2wWzQlwzoyc'
consumer_secret = 'rMupiaWZH7UHbCrkJPmxwYwEoLYs4MrzUBmjYFK0IgDJGDk7RM'
access_token = '769192288111226880-yMvHoVBupxFVInHOQPH8v2AWb3KVTNX'
access_token_secret = 'qu2hYKNcHPAQYBiMaiSjGHGQSSdUY7PC9r4EyTeVSv4UI'


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def collect(twt):
    d = {}
    count = 0
    request = robust_request(twt, 'search/tweets', {'q': '@gucci', 'lang': 'en', 'count': 100})
    for i in request:
        #print(i)
        print("Collecting Tweets"+str(count))
        d[count] = i
        count += 1
    #print("-----")
    #print("len",len(d))
    for i in range(49):
        request = robust_request(twt, 'search/tweets', {'q': '@gucci', 'lang': 'en', 'count': 100, 'max_id' : d[count-1]['id']})
        for j in request:
            d[count] = j
            print("Collecting Tweets" + str(count))
            count += 1
    print("Total number of tweets collected", len(d))

    friends = defaultdict(list)
    tweet_list = []
    for k,v in d.items():
        tweet_list.append(v)
    #print("Len of dictionary", len(d))
    count = 0
    for tweet in tweet_list[:10]:
        #request = robust_request(twt,'followers/ids',{'screen_name':tweet['user']['screen_name']})
        request = robust_request(twt, 'friends/list', {'screen_name': tweet['user']['screen_name'], 'count': 200})
        for r in request:
            friends[tweet['user']['screen_name']].append(r['screen_name'])
        print(count)
        count += 1
    print(friends)
    f = open('Friends.txt', 'w')
    for k,v in friends.items():
        for i in v:
            f.write(str(k)+"\t"+str(i)+"\n")

    f.close()






    data = open('Data.txt','wb')
    pickle.dump(d,data)
    data.close()
    '''
    data= open('Data.txt','rb')
    c = pickle.load(data)
    print((d==c))
    '''

        
    

def main():
    twt = get_twitter()
    collect(twt)

if __name__ == '__main__':
    main()

