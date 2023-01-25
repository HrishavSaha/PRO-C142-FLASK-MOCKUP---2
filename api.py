from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df1 = pd.read_csv('priSA.csv')
demographDF = pd.read_csv("top20.csv")

title = df1['title'].tolist()
contentID = df1['contentId'].tolist()
titles = list(zip(title, contentID))

top20Title = demographDF['title'].tolist()

likedTitle = []
notLikedTitle = []

CV = CountVectorizer(stop_words="english")
count_matrix = CV.fit_transform(df1['title'])
CS = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df1.index, index = list(df1["contentId"]))

def getreccs(cID):
  idx = indices[cID]
  sim_score = list(enumerate(CS[idx]))
  sim_score = sorted(sim_score, key=lambda x:x[1], reverse=True)
  sim_score = sim_score[1:11]
  recc_indices = [i[0] for i in sim_score]
  return df1['contentId'].iloc[recc_indices]

app = Flask(__name__)

@app.route('/getTitles')
def getTitles():
  return jsonify({
    'data':titles,
    'status':'Success'
  }, 200)

@app.route('/liked')
def liked():
  likedTitleName = titles.pop(0)
  likedTitle.append(likedTitleName)
  return jsonify({
    'data':likedTitle,
    'status':'Success'
  }, 200)

@app.route('/notLiked')
def notLiked():
  notLikedTitleName = titles.pop(0)
  notLikedTitle.append(notLikedTitleName)
  return jsonify({
    'data':notLikedTitle,
    'status':'Success'
  }, 200)

@app.route('/top20')
def top20():
  return jsonify({
    'data':top20Title,
    'status':'Success'
  }, 200)

@app.route('/getrecc')
def getrecc():
  cID = int(request.args.get('cID'))
  reccs = getreccs(cID).tolist()
  return jsonify({
    'data':reccs,
    'status':'Success'
  }, 200)

if __name__ == '__main__':
  app.run()