from pyspark.context import SparkContext
import json
from datetime import datetime
import pytz
from xgboost import XGBRegressor
import time
import sys

start_time = time.time()

sc = SparkContext()
sc.setLogLevel('ERROR')

# FOLDER_PATH = sys.argv[1]
# TESTING_FILE_PATH = sys.argv[2]
# OUTPUT_FILE_PATH = sys.argv[3]

FOLDER_PATH = '/Users/veersingh/Desktop/competition_files/'
TESTING_FILE_PATH = '/Users/veersingh/Desktop/competition_files/yelp_val.csv'
OUTPUT_FILE_PATH = '/Users/veersingh/Desktop/Recommendation-System-to-predict-Yelp-ratings/output.csv'

TRAIN_FILE_PATH = FOLDER_PATH + 'yelp_train.csv'
BUSINESS_FILE_PATH = FOLDER_PATH + 'business.json'
CHECKIN_FILE_PATH = FOLDER_PATH + 'checkin.json'
PHOTO_FILE_PATH = FOLDER_PATH + 'photo.json'
TIP_FILE_PATH = FOLDER_PATH + 'tip.json'
USER_FILE_PATH = FOLDER_PATH + 'user.json'



# Read in the training dataset. Remove the header and convert a csv string into a list of 3 elements
# [user_id, business_id, rating(float type)]
train_RDD = sc.textFile(TRAIN_FILE_PATH)
headers_train = train_RDD.first()
train_RDD = train_RDD.filter(lambda x:x!=headers_train).map(lambda x:x.split(',')).map(lambda x:[x[0], x[1], float(x[2])])

#----------- Functions for feature extraction
def get_latitude(latitude_value):
    if not latitude_value:
        return 0
    return latitude_value

def get_longitude(longitude_value):
    if not longitude_value:
        return 0
    return longitude_value

def get_num_attributes(attributes_dict):
    if not attributes_dict:
        return 0
    return len(attributes_dict)

def get_rate_true_attributes(attributes_dict):
    if not attributes_dict:
        return 0
    num_total = 0
    num_true = 0
    for k,v in attributes_dict.items():
        if v in ('True', 'False'):
            num_total += 1
            if v == 'True':
                num_true += 1
    if num_total == 0:
        return 0
    return num_true/num_total
            
def get_num_categories(categories):
    if not categories:
        return 0
    categories = categories.split(',')
    return len(categories)

def get_num_checkins(checkin_data):
    return sum(checkin_data.values())

def get_yelping_since(yelping_since):
    date_obj = datetime.strptime(yelping_since, '%Y-%m-%d')
    utc_date = pytz.utc.localize(date_obj)
    return int(utc_date.timestamp())

def get_num_friends(friends):
    if friends == 'None':
        return 0
    friends = friends.split(',')
    return len(friends)

def get_num_elites(elite):
    if elite == 'None':
        return 0
    elite = elite.split(',')
    return len(elite)

#---------------------------------------------

# Get the following features for each business: id, latitude, longitude, stars, review_count, if its open or closed, rate of true attributes i.e. num true attributes/total attributes and number of categories
business_RDD = sc.textFile(BUSINESS_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'],
                                                                                              [float(get_latitude(x['latitude'])),
                                                                                              float(get_longitude(x['longitude'])),
                                                                                              float(x['stars']),
                                                                                              int(x['review_count']),
                                                                                              int(x['is_open']),
                                                                                              get_rate_true_attributes(x['attributes']),
                                                                                              get_num_categories(x['categories'])]
                                                                                          ))

# Get the total number of check ins for a business
checkIn_RDD = sc.textFile(CHECKIN_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], get_num_checkins(x['time']))).map(lambda x: (x[0], [x[1]]))

# Get the total number of photos for a business
photo_RDD = sc.textFile(PHOTO_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))

# Get the total number of tips given by a user and the total number of tips for each business
tip_RDD = sc.textFile(TIP_FILE_PATH).map(lambda x: json.loads(x))

tips_business_RDD = tip_RDD.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))
tips_user_RDD = tip_RDD.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0], [x[1]]))

# Get the features for each user
user_RDD = sc.textFile(USER_FILE_PATH).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],
                                                                               [
                                                                                   int(x['review_count']),
                                                                                   get_yelping_since(x['yelping_since']),
                                                                                   get_num_friends(x['friends']),
                                                                                   int(x['useful']),
                                                                                   int(x['funny']),
                                                                                   int(x['cool']),
                                                                                   int(x['fans']),
                                                                                   get_num_elites(x['elite']),
                                                                                   float(x['average_stars']),
                                                                                   int(x['compliment_hot']),
                                                                                   int(x['compliment_more']),
                                                                                   int(x['compliment_profile']),
                                                                                   int(x['compliment_cute']),
                                                                                   int(x['compliment_list']),
                                                                                   int(x['compliment_note']),
                                                                                   int(x['compliment_plain']),
                                                                                   int(x['compliment_cool']),
                                                                                   int(x['compliment_funny']),
                                                                                   int(x['compliment_writer']),
                                                                                   int(x['compliment_photos'])
                                                                               ]))


#----------- Create train X and Y
def combine_lists(data_row):
    # fix nonetype error
    if data_row[1][1] == None:
        return[data_row[0], data_row[1][0] + [0]]
    if type(data_row[1][0]) == str:
        return [data_row[0], [data_row[1][0]] + data_row[1][1]]
    return [data_row[0], data_row[1][0] + data_row[1][1]]

# Combine the following RDDs to create a vector for each business with business id as key and list of features as value
# business_RDD + checkIn_RDD + photo_RDD + tips_business_RDD
# make sure to fix NoneType error when combining lists since some values are None
business_features_RDD = business_RDD.leftOuterJoin(checkIn_RDD).map(lambda x: combine_lists(x)).leftOuterJoin(photo_RDD).map(lambda x: combine_lists(x)).leftOuterJoin(tips_business_RDD).map(lambda x: combine_lists(x))


# Combine the following RDDs to create a vector for each user with user id as key and list of features as value
# user_RDD + tips_user_RDD
# make sure to fix NoneType error when combining lists since some values are None
user_features_RDD = user_RDD.leftOuterJoin(tips_user_RDD).map(lambda x: combine_lists(x))

def switch_keys(data_row):
    bus_id = data_row[0]
    usr_id = data_row[1][0]
    features = data_row[1][1:]
    
    return (usr_id, [bus_id] + features)

def join_all(data_row):
    usr_id = data_row[0]
    bus_id = data_row[1][0][0]
    bus_features = data_row[1][0][1:]
    usr_features = data_row[1][1]
    
    return ((usr_id, bus_id), bus_features + usr_features)

# join the train_RDD and business_features_RDD
# we need to have the business_id as the key for this
train_RDD_tmp = train_RDD.map(lambda x: (x[1], x[0]))
train_join_business_features_RDD = train_RDD_tmp.leftOuterJoin(business_features_RDD).map(lambda x: combine_lists(x))

# now join this with the user_features_RDD. We need to have the user_id as key for this
train_join_business_features_RDD_tmp = train_join_business_features_RDD.map(lambda x: switch_keys(x))
train_join_business_features_user_features_RDD = train_join_business_features_RDD_tmp.leftOuterJoin(user_features_RDD)

# format the data as (user_id, business_id) [feature1, feature2, ...]
train_all_joined_MAP = train_join_business_features_user_features_RDD.map(lambda x: join_all(x)).collectAsMap()

# get the values in trainRDD
labels_MAP = train_RDD.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

# create the x and y training lists
x_train = []
y_train = []

for k in train_all_joined_MAP:
    x_train.append(train_all_joined_MAP[k])
    y_train.append(labels_MAP[k])

#----------- Training Phase -----------
# model = XGBRegressor(n_estimators=1000, learning_rate=0.1,verbocity=3, max_depth=5, subsample=0.85, colsample_bytree = 0.9)
model = XGBRegressor(max_depth=5, min_child_weight=1, subsample=0.6, colsample_bytree=0.6, gamma=0, reg_alpha=1, reg_lambda=0, learning_rate=0.05, n_estimators=800)
model.fit(X=x_train, y=y_train, eval_metric = 'rmse')
#--------------------------------------

#----------- Testing Phase -----------
# Read in the testing dataset. Remove the header and convert a csv string into a list of 2 elements
# [user_id, business_id]
test_RDD = sc.textFile(TESTING_FILE_PATH)
headers_test = test_RDD.first()
test_RDD = test_RDD.filter(lambda x:x!=headers_test).map(lambda x:x.split(',')).map(lambda x:(x[0], x[1]))

# join the test_RDD and business_features_RDD
# we need to have the business_id as the key for this
test_RDD_tmp = test_RDD.map(lambda x: (x[1], x[0]))
test_join_business_features_RDD = test_RDD_tmp.leftOuterJoin(business_features_RDD).map(lambda x: combine_lists(x))

# now join this with the user_features_RDD. We need to have the user_id as key for this
test_join_business_features_RDD_tmp = test_join_business_features_RDD.map(lambda x: switch_keys(x))
test_join_business_features_user_features_RDD = test_join_business_features_RDD_tmp.leftOuterJoin(user_features_RDD)

# format the data as (user_id, business_id) [feature1, feature2, ...]
test_all_joined_MAP = test_join_business_features_user_features_RDD.map(lambda x: join_all(x)).collectAsMap()

# create the x testing list
x_test = []
test_labels = []
for k in test_all_joined_MAP:
    x_test.append(test_all_joined_MAP[k])
    test_labels.append(k)
#--------------------------------------

#----------- Predictions -----------
predictions = model.predict(data=x_test)

fhand = open(OUTPUT_FILE_PATH, 'w')
fhand.writelines('user_id, business_id, prediction\n')
for i in range(len(test_labels)):
    fhand.writelines(test_labels[i][0] + ',' + test_labels[i][1] + ',' + str(predictions[i]) + '\n')
fhand.close()

end_time = time.time()
print(f'Duration: {end_time - start_time}')

