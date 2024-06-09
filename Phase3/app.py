import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import warnings 
warnings.filterwarnings('ignore')
hotelbook_file=pd.read_csv('C:/Users/Sahithi Bakaram/Desktop/DIC Project/Phase3/hotel_cleaned.csv')
hotelbook_dataframe=pd.DataFrame(hotelbook_file)
hotelbook_dataframe=hotelbook_dataframe.drop(['reservation_status','country','reservation_status_date','arrival_date','reservation_status'], axis=1)
numeric_columns=['lead_time', 'arrival_date_year','arrival_date_week_number', 'arrival_date_day_of_month','stays_in_weekend_nights', 'stays_in_week_nights', 'adults','is_repeated_guest', 'previous_cancellations','previous_bookings_not_canceled', 'booking_changes','days_in_waiting_list', 'adr', 'required_car_parking_spaces','total_of_special_requests', 'total_stays','total_guests', 'kids']
categorical_columns=['hotel', 'meal', 'market_segment','distribution_channel', 'reserved_room_type', 'assigned_room_type','deposit_type', 'customer_type']
X=hotelbook_dataframe[['hotel', 'meal', 'market_segment','deposit_type', 'customer_type', 'required_car_parking_spaces', 'previous_cancellations', 'booking_changes','total_of_special_requests', 'total_stays','lead_time','total_guests']]
X_numeric_columns=['lead_time','previous_cancellations', 'booking_changes','total_of_special_requests', 'total_stays','total_guests']
#converting to categorical features
X['hotel'].replace(['Resort Hotel','City Hotel'],[0,1],inplace=True)
X['meal'].replace(['BB' , 'FB', 'HB', 'SC', 'Undefined'],[0,1,2,3,4],inplace=True)
X['market_segment'].replace(['Direct', 'Corporate', 'Online TA', 'Offline TA/TO','Complementary', 'Groups', 'Undefined', 'Aviation'],[0,1,2,3,4,5,6,7],inplace=True)
#X['distribution_channel'].replace(['Direct', 'Corporate', 'TA/TO', 'Undefined','GDS'],[0,1,2,3,4],inplace=True)
X['deposit_type'].replace(['No Deposit','Refundable', 'Non Refund'],[0,1,2],inplace=True)
X['customer_type'].replace(['Transient','Contract','Transient-Party','Group'],[0,1,2,3],inplace=True)
#predictor variable
Y=hotelbook_dataframe['is_canceled']
# normalize numeric features.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[X_numeric_columns] = scaler.fit_transform(X[X_numeric_columns])
# separating training data and testing data.
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
# Fitting  RandomForestClassifier to the Training se
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=18,criterion='gini', max_features = 'sqrt', n_jobs=-1, verbose=1, random_state=0)
model.fit(XTrain, YTrain)
# Predicting the Test set results
YPred = model.predict(XTest)
import pickle
# Saving model to disk
pickle.dump(model, open('C:/Users/Sahithi Bakaram/Desktop/DIC Project/Phase3/model.pkl','wb'))

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('C:/Users/Sahithi Bakaram/Desktop/DIC Project/Phase3/model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.form["hotel"] == 'City Hotel':
        hotel = 1
    else:
        hotel = 0
        
    if request.form["meal"] == 'BB':
        meal = 0
    elif request.form["meal"] == 'FB':
        meal = 1
    elif request.form["meal"] == 'HB':
        meal = 2
    elif request.form["meal"] == 'SC':
        meal = 3
    else:
        meal = 4
        
    if request.form["marketsegment"] == 'Direct':
        marketsegment = 0
    elif request.form["marketsegment"] == 'Corporate':
        marketsegment = 1
    elif request.form["marketsegment"] == 'Online TA':
        marketsegment = 2
    elif request.form["marketsegment"] == 'Offline TA/TO':
        marketsegment = 3
    elif request.form["marketsegment"] == 'Complementary':
        marketsegment = 4
    elif request.form["marketsegment"] == 'Groups':
        marketsegment = 5
    elif request.form["marketsegment"] == 'Aviation':
        marketsegment = 7
    else:
        marketsegment = 6
        
    if request.form["deposittype"] == 'No Deposit':
        deposittype = 0
    elif request.form["deposittype"] == 'Refundable':
        deposittype = 1
    else:
        deposittype = 2

    if request.form["customertype"] == 'Transient':
        customertype = 0
    elif request.form["customertype"] == 'Transient Party':
        customertype = 2
    elif request.form["customertype"] == 'Contract':
        customertype = 1
    else:
        customertype = 3
    
    if request.form["carparking"] == 'Yes':
        carparking =1
    else:
        carparking = 0
    
    pc = int(request.form["previouscancellations"])
    tg = int(request.form["totalguests"])
    ts = int(request.form["totalstays"])
    lt = int(request.form["leadtime"])
    sr = int(request.form["specialrequests"])
    bc = int(request.form["bookingchanges"])
    d = {'previous_cancellations':pc, 'booking_changes':bc,'total_of_special_requests':sr, 'total_stays':ts,'lead_time':lt,'total_guests':tg}
    test = pd.DataFrame(data = d,index = [0])
    test[X_numeric_columns] = scaler.transform(test[X_numeric_columns])
    print(test)
    test['hotel'] = hotel
    test['meal'] = meal
    test['market_segment'] = marketsegment
    test['deposit_type'] = deposittype
    test['customer_type'] = customertype
    test['required_car_parking_spaces'] = carparking
    prediction = model.predict(test)
    prediction_prob = model.predict_proba(test)
    prediction_confirmed = round(100 * prediction_prob[0][0],2)
    prediction_cancelled = round(100 * prediction_prob[0][1],2)
    if prediction[0] == 0:
        pred_text = "CONFIRMED"
    else:
        pred_text = 'CANCELLED'
    return render_template('result.html', prediction_text=pred_text,round_prediction_cancelled = prediction_cancelled, round_prediction_confirmed = prediction_confirmed)
if __name__ == "__main__":
    app.run(debug=True)
