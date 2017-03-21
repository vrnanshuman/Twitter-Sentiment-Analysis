from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import tweepy
import re_module as s

import socket
socket.getaddrinfo('localhost', 8080)

# import re_module as s 

#consumer key, consumer secret, access token, access secret.
ckey="dv50WXn1qLSuIRxlcgHJAndkE"
csecret="bvjJjxxODdp5tXYbLqM76IkKaoUgq36wleUWo95DL5kCEpdTZh"
atoken="738497117057130496-AWSqJUA2qNJsOEAftxzCSxGb1sT3g6f"
asecret="0MargfKPbfsrykczju8GBdJOFDXnSvpn1aa26Nr1MkI7x"

class listener(StreamListener):

    def on_data(self, data):
        try:
           all_data = json.loads(data)
           tweet = all_data["text"]
           sentiment_value, confidence = s.sentiment(tweet)
           print(tweet, sentiment_value, confidence)
           # print all_data
           if confidence*100 >= 80:
               # print tweet
               # print "\n"
               output = open("twitter_result2.txt","a")
               output.write(sentiment_value)
               output.write('\n')
               output.close()
        except Exception as e:
            pass
        return True

    def on_error(self, status):
        # if(status==420):
        #     return False
        print(status)
        return True

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=['Virat Kohli'],async=True)