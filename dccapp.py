import flask
from flask import Flask,request, render_template, jsonify, url_for, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET'])

def home():
        return render_template(["dcc2.html","dcc.css"])
@app.route("/predict", methods=['POST'])
def upload():
    if request.method == 'POST':
        urls = request.form["Url"]
        #return render_template("ab.html")
        comment_list = []
        def ScrapComment(url):
            option = webdriver.ChromeOptions()
            option.add_argument("--headless")
            driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=option)
            driver.get(url)
            prev_h = 0
            while True:
                height = driver.execute_script("""
                        function getActualHeight() {
                            return Math.max(
                                Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                                Math.max(document.body.offsetHeight, document.documentElement.offsetHeight),
                                Math.max(document.body.clientHeight, document.documentElement.clientHeight)
                            );
                        }
                        return getActualHeight();
                    """)
                driver.execute_script(f"window.scrollTo({prev_h},{prev_h + 200})")
                # fix the time sleep value according to your network connection
                time.sleep(1)
                prev_h +=200  
                if prev_h >= height:
                    break
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
            comment_div = soup.select("#content #content-text")
            
            for x in comment_div:
                comment_list.append(x.text)
            
            #data = pd.DataFrame({'coments':comment_list})
            #data.to_csv('C:\\Users\\ADMIN\\Downloads\\ytcomments.csv')
                
        ScrapComment(urls)

        df = pd.read_csv("Tweets.csv")
        tweet_df = df[['text','airline_sentiment']]
        tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
        sentiment_label = tweet_df.airline_sentiment.factorize()
        tweet = tweet_df.text.values
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(tweet)
        vocab_size = len(tokenizer.word_index) + 1
        encoded_docs = tokenizer.texts_to_sequences(tweet)
        padded_sequence = pad_sequences(encoded_docs, maxlen=200)

        model = load_model("Analysis.h5")
        def predict_sentiment(text):
            tw = tokenizer.texts_to_sequences([text])
            tw = pad_sequences(tw,maxlen=200)
            prediction = int(model.predict(tw).round().item())
            return prediction
            
        x=0
        y=0
        for i in comment_list:
            y_pred=predict_sentiment(i)
            if y_pred==0:
                x +=1
            else:
                y +=1

        m = len(comment_list)
        g=(x*100)/m
        result = "Positive comments in percentage: " + str(g) + "%" + "     "

        g1=(y*100)/m
        result += "Negative comments in percentage: " + str(g1) + "%"
        print(result)
        date = result
        return redirect(url_for('ab', date=date))
    return render_template('dcc2.html')

@app.route('/ab')
def ab():
    date = request.args.get('date', None)
    return render_template('ab.html', date=date)    

if __name__ == "__main__":
    app.run(debug=True)
