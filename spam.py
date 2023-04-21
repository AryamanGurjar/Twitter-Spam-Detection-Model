from flask import Flask, render_template, request
import pickle
import pandas as pd

import tweepy
import csv


auth = tweepy.OAuthHandler("BRNfDkhG8wmQvfhagfjhD2ksN", "tQ0T77HVc62ZBHomipcL3zcYtNKA8KNMC3xkWd5YotuNQhVc9n")
auth.set_access_token("1359887234866966530-aKJjdA4peAVPppGtpEmFOb6HVrTMLk", "ijwTGUyzBaa59E0uS7F3sEcUrCWAu9Lqr5hoWKYi81OAH")


api = tweepy.API(auth)





# Load the vectorizer and the model
cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        hashtag = request.form['hashtag']
        tweets = tweepy.Cursor(api.search_tweets, hashtag).items(100)
        # f = request.files['file']
        # df = pd.read_excel(f)
        # messages = df['message'].tolist()
        # Convert the text to a numerical representation
        # messages = df['message'].astype(str).dropna().tolist()
        # messages = cv.transform(messages)
        mess_tweet = []
        pred= []
        for tweet in tweets:
           message = tweet.text
           mess_tweet.append(message)
           prediction = [message]
        # print(prediction)
        # # Convert the text to a numerical representation
           vector = cv.transform(prediction).toarray()
        # # Make the prediction
           my_pred = model.predict(vector)
       # return render_template('result.html', prediction=my_pred)
        # predictions = model.predict(messages)
        # Add the predictions as a new column in the dataframe
           
           if(my_pred == 1):
             pred.append('Spam')
           elif(my_pred== 0):
             pred.append('Not Spam')
        # Convert the dataframe to HTML
        tweet_dict = {'tweet': mess_tweet, 'prediction': pred}
        
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(tweet_dict)
        
        # Convert the DataFrame to a map object
        df_map = map(dict, df.to_dict('records'))
        
        # Render the result template and pass the result map object as a parameter
        return render_template('result.html', result_map=df_map)

if __name__ == '__main__':
    app.run(debug=True)
