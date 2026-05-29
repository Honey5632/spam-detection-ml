import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

try:
    while True:
        msg = input("Enter message (or type 'exit' to quit): ")
        if msg.lower() == "exit":
            break
        msg_tfidf = vectorizer.transform([msg])
        pred = model.predict(msg_tfidf)[0]
        prob = model.predict_proba(msg_tfidf)[0]

        print("Prediction:", "SPAM" if pred == 1 else "NOT SPAM")
        print("Spam probability:", round(prob[1] * 100, 2), "%")
except KeyboardInterrupt as e:
    print(e)