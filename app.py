from flask import Flask, request
from flask import render_template

import pickle

app = Flask(__name__)

#chargement des modèles
file1 = open("static/models/count_vectorizer_model.sav", 'rb')
vectorizer = pickle.load(file1)

file2 = open("static/models/logistic_regression_model.sav", 'rb')
lr = pickle.load(file2)

@app.route("/api/spamdetector", methods=["POST", "GET"])
def detector():

    if request.method == 'POST':
        mail = request.form['mail']
  
        mail_2 = vectorizer.transform([mail]).toarray()
        p = lr.predict_proba(mail_2.reshape(1, -1))[0]

        ham, spam = p[0], p[1]

        if spam > ham:
            message = "Ce mail est un spam à " + str(spam * 100) + " %."
        
        else:
            message = "Ce mail est un ham à " + str(ham * 100) + " %."

        return render_template("home.html", message=message, mail=mail)

    else:
        return render_template("home.html")
    

if __name__ == "__main__":
    app.run(port=8001, debug=True, host="127.0.0.1")