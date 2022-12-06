from flask import Flask, render_template, jsonify, request
import cryptocompare
import pandas as pd 
from datetime import datetime, timedelta
import time
from newsapi import NewsApiClient
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.backend import shape
from keras.models import load_model


app = Flask(__name__)

filemap = {'btc': 'BTC.h5', 'eth':'ETH.h5', 'ada':'ADA.h5', 'xrp':'XRP.h5', 'avax':'AVAX.h5', 'bnb':'BNB.h5', 'dai':'DAI.h5', 'doge':'DOGE.h5', 'dot':'DOT.h5', 'link':'LINK.h5', 'matic':'MATIC.h5', 'shib':'SHIB.h5', 'uni':'UNI.h5', 'sol':'SOL.h5'}
newsapi = NewsApiClient(api_key='23c0dbae53ec48478d732c848adbc160')
sc=MinMaxScaler()

def editDF(test_df):
  test_df.sort_values(by = 'time', ascending = True, inplace = True)
  test_df.drop(['conversionType', 'conversionSymbol', 'volumefrom'], axis=1, inplace=True)
  test_df.reset_index(drop = True, inplace = True)
  test_df.columns = ['Date','High','Low','Open', 'Volume', 'Close']
  #test_df.set_index('Date', inplace = True)
  test_df.Date = pd.to_datetime(test_df.Date, unit = 's')

def analyzeText(text=None, url=None):
  '''
  This function gets as input either a piece of text or a URL
  and then contacts the IBM Watson NLU API to perform sentiment
  and emotion analysis on the text/url.
  '''
  #My Credential for Watson Natural Language Processing API
  IBM_SERVER_URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/55c5be0c-2c4c-494d-b203-f8ea3154a608'
  IBM_API_KEY = '9txKuixO_PkpR-bAKb-3DWLucTFxdE7l12LGhqLTtweg'

  endpoint = f"{IBM_SERVER_URL}/v1/analyze"
  username = "apikey"
  password = IBM_API_KEY
  
  parameters = {
      'features': 'sentiment',
      'version' : '2022-04-07',
      'text': text,
      'language' : 'en',
      'url' : url # this is an alternative to sending the text
  }

  resp = requests.get(endpoint, params=parameters, auth=(username, password))
  
  error_check = resp.json()

  if 'error' in error_check:
    return None
  else:
    return resp.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def pred():
    return render_template('prediction.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/getnews')
def getnews():
    param = (request.args.get('query')) 

    req_headlines = newsapi.get_everything(q = param,
        language='en',
        page_size=10)

    arr = req_headlines['articles']
    sentiment_score = []

    for item in arr:
        score = analyzeText(url = item['url'])
        sentiment_score.append(score)

    return jsonify({'news_article':req_headlines, 'score':sentiment_score})
    
@app.route('/getdata')
def getdata():
    #Receive Params
    
    param = (request.args.get('coin_name')).upper() 

    if((cryptocompare.get_price(param)) is None):
        return(jsonify({'data': 'Error'}))
   


    current_date = time.mktime((datetime.now()).timetuple()) 
    dt = datetime.now()
    req_date = dt - timedelta(days=200)
    test_df = pd.DataFrame()

    while (dt > req_date):
        temp = pd.DataFrame(cryptocompare.get_historical_price_hour(param, 'USD', limit=1644, exchange='CCCAGG', toTs=dt))
        test_df = test_df.append(temp)
        current_date = current_date - (3600*1644)
        dt = (datetime.fromtimestamp(current_date - (3600*1644)))
    
    editDF(test_df)
    result = test_df.to_json(orient="split", index=False)

    return result

@app.route('/getpred')
def getpred():
    param = (request.args.get('coin_name'))

    dt = datetime.now()
    test_df = pd.DataFrame()
   
    temp = pd.DataFrame(cryptocompare.get_historical_price_hour(param, 'USD', limit=72, exchange='CCCAGG', toTs=dt))
    test_df = test_df.append(temp)
    editDF(test_df)

    TimeSteps = 12
    FutureTimeSteps = 1

    last_datetime = test_df['Date'][test_df.index[-1]]
    unix_datetime =  (time.mktime(last_datetime.timetuple()))

    close_data = test_df["Close"].values.reshape(-1,1) 
    mod_df = test_df[['Date', 'Close']]
    
    DataScaler = sc.fit(close_data)
    regressor = load_model('./models/' + filemap[param])
    last_ten_prices = (close_data[-TimeSteps:])
    next_day_pred = []
    cnt = 0
    pred_date = 0

    while cnt != 6:
        transformed_ten_prices = DataScaler.transform(np.array(last_ten_prices).reshape(-1,1))
        transformed_ten_prices = transformed_ten_prices.reshape(1, TimeSteps, FutureTimeSteps)
        next_day_price = regressor.predict(transformed_ten_prices)
        next_day_price = DataScaler.inverse_transform(next_day_price)
        next_day_pred.append(next_day_price)
        cnt += 1
        last_ten_prices = last_ten_prices[1:]
        pred_date = unix_datetime + (3600*cnt)
        mod_df.loc[len(mod_df.index)] = ([datetime.fromtimestamp(pred_date), next_day_price[0][0]])
        last_ten_prices = np.insert(last_ten_prices, 11, next_day_price)

    result = mod_df.to_json(orient="split", index=False)
    return result
    

if __name__ == "__main__":
    app.run(debug=True)


