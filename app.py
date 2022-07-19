import flask
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

# Use pickle to load in the pre-trained model.
with open(f'model/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
    
with open(f'model/classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)
    

    
with open(f'model/applause_regressor.pkl', 'rb') as file:
    applause_regressor = pickle.load(file)

def predict_category(text):
    result = classifier.predict(tfidf_vectorizer.transform([text]))
    return(result[0])

parties = ['90/Die', 'LINKE', 'FDP', 'CDU/CSU', 'SPD', 'AfD']
    
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        news_text = ' '
        news_text = flask.request.form['news_text']
        prediction = predict_category(news_text)
        
        applause_value = applause_regressor.predict(["news_text"])
        abc = []
        for p,applause in zip(parties,applause_value.T):
            abc.append(f'applause_predicted-{p} ='+ str(applause))
        return flask.render_template('main.html',
                                     original_input={'News Text':news_text},
                                     result=[prediction, abc])


    
if __name__ == '__main__':
    app.run()
