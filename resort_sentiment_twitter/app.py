from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import os
import tweepy as tweepy
import pandas as pd
# import HashingVectorizer from local dir
from vectorizer import vect

#need to hide this token in a config file
client = tweepy.Client(bearer_token='enter the token')


app = Flask(__name__)


######## Prepare the Classifier (load pickle)
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'tweets.sqlite')

######## Get live tweets using twitter api
def _get_client():
    #client and token should be hidden
    return tweepy.Client(bearer_token='')

def get_tweets(skiresort):
    query = f'{skiresort} has:images lang:en -is:retweet'
    tweets = _get_client().search_recent_tweets(query=query, \
             tweet_fields=['context_annotations', 'created_at'], max_results=10)
    live_data = []
    for tweet in tweets.data:
        #print('text', tweet.text, ' \n ', 'created',tweet.created_at)
        live_data.append((tweet.text, tweet.created_at))

    return pd.DataFrame(live_data, columns=['tweet', 'created'])


######## Classify the list of live tweets
def classify(skiresort, clf):

    #get live tweets for skiresort entered
    live_tweets = get_tweets(skiresort)

    #convert to an array
    #vect is loaded at top in global
    X = vect.transform(live_tweets['tweet'].values)

    #assign labels for 0/1
    label = {0: 'negative', 1: 'positive'}

    #make sentiment prediction for each tweet and add to df to print later
    live_tweets['sentiment prediction'] = clf.predict(X)
    live_tweets['probability'] = [max(i) for i in clf.predict_proba(X)]

    #return the mean probability of prediction (take max value)
    prediction_result = clf.predict_proba(X)
    prediction_result_meanvals = [np.mean(prediction_result[:,0]),\
                                  np.mean(prediction_result[:,1])]
    proba = np.max(prediction_result_meanvals)
    y_pred = prediction_result_meanvals.index(proba)
    return label[y_pred], proba, live_tweets


### Update model and store twitter data in db (not implemented)
'''def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()'''

######## Flask
class ReviewForm(Form):
    skiresort = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=5)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        skiresort = request.form['skiresort']
        y, proba, df_tweets = classify(skiresort, clf)
        return render_template('results.html',
                                content=skiresort,
                                prediction=y,
                                probability=round(proba*100, 2),
                                tables = [df_tweets.to_html(index=False,classes=skiresort)]
                                )
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])

def feedback():
    feedback = request.form['feedback_button']
    skiresort = request.form['skiresort']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]

    #update model with feedback (not implemented)
    #if feedback == 'Incorrect':
    #    y = int(not(y))
    #train(review, y)
    #sqlite_entry(db, review, y)

    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(host="localhost", port=8500, debug=True)
