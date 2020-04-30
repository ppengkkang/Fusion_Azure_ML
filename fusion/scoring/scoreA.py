import joblib

def init():
    global model
    model = joblib.load("../models/model.pkl")


def run():
    preds = model.predict([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]])
    print(preds)
    return preds

if __name__ == "__main__":
    init()
    run()

