from pyspark.context import SparkContext
import json
import pytz
from datetime import datetime
from xgboost import XGBRegressor
import time
import sys

start_time = time.time()

sc = SparkContext()
sc.setLogLevel('ERROR')

FOLDER_PATH = sys.argv[1]
INPUT_FILE_PATH = FOLDER_PATH + '/yelp_train.csv'
USER_DATASET_PATH = FOLDER_PATH + '/user.json'
BUSINESS_DATASET_PATH = FOLDER_PATH + '/business.json'
TESTING_FILE_PATH = sys.argv[2]
OUTPUT_FILE_PATH = sys.argv[3]

# FOLDER_PATH = '/Users/veersingh/Desktop/hw3_downloaded_files/'
# INPUT_FILE_PATH = FOLDER_PATH + '/yelp_train.csv'
# USER_DATASET_PATH = FOLDER_PATH + '/user-001.json'
# BUSINESS_DATASET_PATH = FOLDER_PATH + '/business.json'
# TESTING_FILE_PATH = '/Users/veersingh/Desktop/hw3_downloaded_files/yelp_val_in.csv'
# OUTPUT_FILE_PATH = '/Users/veersingh/Desktop/Recommendation-System-with-XGBoost-and-Spark/output2_2.csv'

# Read in the training dataset. Remove the header and convert a csv string into a list of 3 elements
# [user_id, business_id, rating(float type)]
train_RDD = sc.textFile(INPUT_FILE_PATH)
headers_train = train_RDD.first()
train_RDD = train_RDD.filter(lambda x:x!=headers_train).map(lambda x:x.split(',')).map(lambda x:[x[0], x[1], float(x[2])])

# Read in the user dataset and convert each json string into dict
user_data_RDD = sc.textFile(USER_DATASET_PATH).map(lambda x: json.loads(x))

def change_yelping_since(x):
    # change yelping since to the age of the account by converting it to utc timestamp
    date_obj = datetime.strptime(x['yelping_since'], '%Y-%m-%d')
    utc_date = pytz.utc.localize(date_obj)
    x['yelping_since'] = int(utc_date.timestamp())
    return x

# change yelping_since to utc timestamp. then extract all features
user_data_RDD = user_data_RDD.map(change_yelping_since).map(lambda x: (x['user_id'], [
    x['review_count'],
    x['yelping_since'],
    x['useful'],
    x['fans'],
    x['average_stars']
]))

# Read in the business dataset and convert each json string into dict
business_data_RDD = sc.textFile(BUSINESS_DATASET_PATH).map(lambda x: json.loads(x))

# Extract features
business_data_RDD = business_data_RDD.map(lambda x: (x['business_id'], [
    x['stars'], 
    x['review_count']
]))


# join the train_RDD and the business_data_RDD
# we need to have the business_id as the key for this
train_RDD_tmp = train_RDD.map(lambda x: (x[1], x[0]))
train_join_business_RDD = train_RDD_tmp.leftOuterJoin(business_data_RDD)
# now join this with the user_data_RDD. We need to have the user_id as key for this
train_join_business_RDD_tmp = train_join_business_RDD.map(lambda x: (x[1][0], [x[0], x[1][1][0], x[1][1][1]]))
train_join_business_join_user_RDD = train_join_business_RDD_tmp.leftOuterJoin(user_data_RDD)
# format the data as (user_id, business_id) [feature1, feature2, ...]
train_all_joined_RDD = train_join_business_join_user_RDD.map(lambda x: ((x[0], x[1][0][0]), 
                                                                        [
                                                                            x[1][0][1],
                                                                            x[1][0][2],
                                                                            x[1][1][0],
                                                                            x[1][1][1],
                                                                            x[1][1][2],
                                                                            x[1][1][3],
                                                                            x[1][1][4]
                                                                        ])).collectAsMap()

labels = train_RDD.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

# create the x and y training lists
x_train = []
y_train = []

for k in train_all_joined_RDD.keys():
    x_train.append(train_all_joined_RDD[k])
    y_train.append(labels[k])

# Train the model
model = XGBRegressor(max_depth=8, n_estimators=400)
model.fit(x_train, y_train)

# Testing Phase
# Read in the testing dataset. Remove the header and convert a csv string into a list of 2 elements
# [user_id, business_id]
test_RDD = sc.textFile(TESTING_FILE_PATH)
headers_test = test_RDD.first()
test_RDD = test_RDD.filter(lambda x:x!=headers_test).map(lambda x:x.split(',')).map(lambda x:(x[0], x[1]))

# join the test_RDD and the business_data_RDD
# we need to have the business_id as the key for this
test_RDD_tmp = test_RDD.map(lambda x: (x[1], x[0]))
test_join_business_RDD = test_RDD_tmp.leftOuterJoin(business_data_RDD)
# now join this with the user_data_RDD. We need to have the user_id as key for this
test_join_business_RDD_tmp = test_join_business_RDD.map(lambda x: (x[1][0], [x[0], x[1][1][0], x[1][1][1]]))
test_join_business_join_user_RDD = test_join_business_RDD_tmp.leftOuterJoin(user_data_RDD)
# format the data as (user_id, business_id) [feature1, feature2, ...]
test_all_joined_RDD = test_join_business_join_user_RDD.map(lambda x: ((x[0], x[1][0][0]), 
                                                                        [
                                                                            x[1][0][1],
                                                                            x[1][0][2],
                                                                            x[1][1][0],
                                                                            x[1][1][1],
                                                                            x[1][1][2],
                                                                            x[1][1][3],
                                                                            x[1][1][4]
                                                                        ])).collectAsMap()

# create the x testing list
x_test = []
test_labels = []
for k in test_all_joined_RDD.keys():
    x_test.append(test_all_joined_RDD[k])
    test_labels.append(k)

predictions = model.predict(data=x_test)

fhand = open(OUTPUT_FILE_PATH, 'w')
fhand.writelines('user_id, business_id, prediction\n')
for i in range(len(test_labels)):
    fhand.writelines(test_labels[i][0] + ',' + test_labels[i][1] + ',' + str(predictions[i]) + '\n')
fhand.close()

end_time = time.time()
print(f'Duration: {end_time - start_time}')