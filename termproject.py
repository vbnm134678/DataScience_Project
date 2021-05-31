from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.core.display import display
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Function for label encoding
def encoder(data):
    for col in data.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
    return data


# A function that calculates the distance between two points
def euclidean_distance(row1, row2):
    distance = 0.0
    # Calculate distance
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()

    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    # Sorting
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    # k nearest neighbors value
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Function to predict the results of test data
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    # for neighbor in neighbors:
    #    print("Nearest row ", neighbor)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Function for LabelEncoder
def makeLabelEncoderData(data):
    encodedData = encoder(
        data.loc[:, ["animal_type", "outcome_subtype", "outcome_type", "sex_upon_outcome"]])

    label = encodedData.loc[:, ["animal_type", "outcome_subtype", "sex_upon_outcome", "outcome_type"]]
    # print(label)
    return label

# Function for OneHotEncoder
def makeOneHotEncoderData(data):
    X = data.loc[:, ["animal_type", "outcome_subtype", "outcome_type", "sex_upon_outcome"]]

    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(X)

    oh = pd.DataFrame(onehot_encoder.transform(X).toarray())
    oh.head()

    return oh

# Function for OrdinalEncoder
def makeOrdinalEncoderData(data):
    ordinal_encoder = OrdinalEncoder()

    X = data.loc[:, ["animal_type", "outcome_subtype", "sex_upon_outcome", "outcome_type"]]
    ordinal_encoder.fit(X)

    ordi = pd.DataFrame(ordinal_encoder.transform(X),
                        columns=["animal_type", "outcome_subtype", "sex_upon_outcome", "outcome_type"])
    ordi.head()
    # print(ordi)
    return ordi

# Apply knn algorithm, predict accuracy
def knn(data_train, data_test):
    d_train = []
    train_x = []
    train_y = []
    train_z = []

    test_x = []
    test_y = []
    test_z = []

    confusion_test = []
    confusion_predict = []
    array_num = []

    # Extract the values from the train data and put them in an array.
    for i in range(len(data_train)):
        temp = data_train.iloc[i]
        d_train.append([temp[0], temp[1], temp[2], temp[3]])
        # Save coordinates to an array to draw a plot of the Scatter
        train_x.append([temp[0]])
        train_y.append([temp[1]])
        train_z.append([temp[2]])

    # Total number of test
    total = 0
    # Estimated value = actual value count
    correct = 0

    number_of_test = len(data_test)
    # number_of_test = 1000
    # for i in tqdm(range(number_of_test)):
    for i in range(number_of_test):
        temp = data_test.iloc[i]
        row = [temp[0], temp[1], temp[2], temp[3]]
        test_x.append([temp[0]])
        test_y.append([temp[1]])
        test_z.append([temp[2]])

        print("\nPredicting row: ", row)
        prediction = predict_classification(d_train, row, 3)



        print('Outcome data is %s, Prediction is %s.' % (row[3], prediction))
        if (row[3] == 3 and prediction == 3):
            confusion_predict.append(1)
            confusion_test.append(1)
            # Get row value of predicted successful value
            array_num = np.append(array_num, np.array([i]))

        if (row[3] == 3 and prediction != 3):
            confusion_predict.append(0)
            confusion_test.append(1)

        if (row[3] != 3 and prediction == 3):
            confusion_predict.append(1)
            confusion_test.append(0)

        if (row[3] != 3 and prediction != 3):
            confusion_predict.append(0)
            confusion_test.append(0)
        # If Predicted value == actual value, correct ++
        if row[3] == prediction:
            correct += 1


        total += 1
    # Print accuracy
    print("Accuracy : {}".format(float(correct) / float(total) * 100))

    draw_graph(train_x, train_y, train_z, test_x, test_y, test_z)
    draw_confusion(confusion_test, confusion_predict)
    return array_num

# Draw a 3D scatter plot
def draw_graph(train_x, train_y, train_z, test_x, test_y, test_z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(train_x, train_y, train_z, marker='o', s=5, c='darkgreen', label="train")
    ax.scatter(test_x, test_y, test_z, marker='^', s=30, c='red', label="test")
    ax.set_xlabel("animal_type", fontsize=6)
    ax.set_ylabel("outcome_subtype", fontsize=6)
    ax.set_zlabel("sex_upon_outcome", fontsize=6)
    ax.set_title("3d Sactter plot in kNN", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")

    plt.show()

# Draw confusion matrix
def draw_confusion(confusion_test, confusion_predict):
    cm = sklearn.metrics.confusion_matrix(confusion_test, confusion_predict)
    print(cm)
    plt.figure(figsize=(2, 2))
    sn.heatmap(data=cm, annot=True, fmt='.2f', linewidths=.1, cmap='Blues')

    plt.show()


# Search data with user criteria
def search(ori_data, test, type, breed, color, array_num):
    test_x = []
    test_y = []
    test_z = []
    result_x = []
    result_y = []
    result_z = []

    number_of_test = len(test)
    for i in range(number_of_test):
        temp = test.iloc[i]
        test_x.append([temp[0]])
        test_y.append([temp[1]])
        test_z.append([temp[2]])

    # Use 'row_num' to find objects that meet the needs of the user
    for i in array_num:
        l = int(i)
        temp2 = ori_data[l]
        count = 0
        if type == temp2[2]:
            count += 1
        if breed == temp2[3]:
            count += 1
        if color == temp2[4]:
            count += 1
        # Print if any of the user requirements are met
        if count > 0:
            if (temp2[2] == 1):
                temp2[2] = 'Cat'
            if (temp2[2] == 2):
                temp2[2] = 'Dog'
            if (temp2[2] == 3):
                temp2[2] = 'Bird'
            if (temp2[2] == 4):
                temp2[2] = 'Other'
            if (temp2[2] == 5):
                temp2[2] = 'Livestock'
            # From the original data 'ori_data', print age, animal type, breed, and fur color
            print('User recommendation : ', temp2[0], ', ', temp2[2], ', ', temp2[3], ', ', temp2[4])

            case = test.iloc[l]
            result_x.append(case[0])
            result_y.append(case[1])
            result_z.append(case[2])

    draw_graph(test_x, test_y, test_z, result_x, result_y, result_z)


def main():
    pd.set_option('display.max_columns', None)
    # read data
    data = pd.read_csv('C:/Users/Lee/Desktop/데이터과학/TermProject/aac_shelter_outcomes.csv')
    # Column description
    data_cols = data.columns
    data.info()

    data.iloc[0]

    # Describe relation between outcome type & subtype
    data.groupby(['outcome_type', 'outcome_subtype']).size()

    data.head()

    # Missing value
    data.isnull().sum()

    # Drop name ( missing value & don't needed )
    data = data.drop(['name'], axis=1)

    # Drop (Aggressive, Medical, Rabies Risk )
    data = data[data['outcome_subtype'] != 'Aggressive']
    data = data[data['outcome_subtype'] != 'Medical']
    data = data[data['outcome_subtype'] != 'Rabies Risk']

    data.groupby(['outcome_type', 'outcome_subtype']).size()
    data.groupby(['age_upon_outcome']).size()

    data.isnull().sum()

    # Fill missing subtype using outcome type value
    data['outcome_subtype'] = data["outcome_subtype"].fillna(data['outcome_type'])

    data.isnull().sum()

    # Drop missing value
    data = data.dropna()
    data.isnull().sum()

    # Save data before encoding
    ori_data = data.to_numpy()

    # Scale the age on a weekly basis.
    # Separate the number from the unit
    data['age_week'] = data['age_upon_outcome'].str[0:1]
    data["age_week"] = data["age_week"].replace("[\$]", "", regex=True)
    data["age_week"] = pd.to_numeric(data["age_week"])

    # Year Extracted values
    # year=data.loc[(data['age_upon_outcome'].str.contains('years'))]
    year = data.loc[(data['age_upon_outcome'].str.contains('year'))].copy()
    year = year

    # Month Extracted values
    # month=data.loc[(data['age_upon_outcome'].str.contains('months'))]
    month = data.loc[(data['age_upon_outcome'].str.contains('month'))].copy()
    month = month

    # Day Extracted values
    # day=data.loc[(data['age_upon_outcome'].str.contains('days'))]
    day = data.loc[(data['age_upon_outcome'].str.contains('day'))].copy()
    day = day

    # week=data.loc[(data['age_upon_outcome'].str.contains('weeks'))]
    week = data.loc[(data['age_upon_outcome'].str.contains('week'))].copy()
    # To meet the weekly basis, the annual unit is multiplied by 52, the monthly unit is multiplied by 4,
    # and the first unit is multiplied by 1/7    year["age_week"] = pd.DataFrame(year["age_week"] * 52)
    month["age_week"] = pd.DataFrame(month["age_week"] * 4)

    day["age_week"] = pd.DataFrame(day["age_week"] * 1 / 7)

    data = pd.concat([year, month, week, day])
    # Rearrange by Index Order
    data.sort_index()

    a = data["age_week"]
    a.values

    from sklearn import preprocessing
    # Standard scaling
    scaler_s = preprocessing.StandardScaler()
    data["ss_age"] = scaler_s.fit_transform(data[["age_week"]])

    # MinMax scaling
    scaler_mm = preprocessing.MinMaxScaler()
    data["mm_week"] = scaler_mm.fit_transform(data[["age_week"]])

    # MaxAbs scaling
    scaler_ma = preprocessing.MaxAbsScaler()
    data["ma_age"] = scaler_ma.fit_transform(data[["age_week"]])

    # Robust scaling
    scaler_r = preprocessing.RobustScaler()
    data["rs_age"] = scaler_r.fit_transform(data[["age_week"]])

    print(data)
    display(sn.countplot(x="outcome_type", data=data, hue="sex_upon_outcome"))
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.title("Results by sex")
    plt.show()
    display(sn.boxplot(x="outcome_type", y="age_week", data=data))
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.title("Results by age(in weeks)")
    plt.show()
    # Rearrange by Index Order
    data.sort_index()
    # print(data)

    # Make custom test data
    custom_test_data = pd.DataFrame([[2, 22, 0, 3], [2, 22, 2, 3], [2, 15, 1, 8], [2, 0, 3, 3]])

    # Make label encoded data
    labelData = makeLabelEncoderData(data)
    # Split dataset
    labelData_train, labelData_test = train_test_split(labelData, test_size=0.2)
    array_num = knn(labelData_train, labelData_test)
    search(ori_data, labelData_test, "Cat", "Domestic", "White", array_num)

    # Make OneHot encoded data
    onehotData = makeOneHotEncoderData(data)

    # Make Ordinal encoded data
    ordinalData = makeOrdinalEncoderData(data)
    # Split dataset
    ordinalData_train, ordinalData_test = train_test_split(ordinalData, test_size=0.2)
    array_num = knn(ordinalData_train, ordinalData_test)
    search(ori_data, ordinalData_test, "Cat", "Domestic", "White", array_num)


main()
