import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC

# Load the csv file
df = pd.read_csv("Traffic.csv")


df['Day of the week'] = df['Day of the week'].replace({'Monday':1,'Tuesday':2,
                                                          'Wednesday':3,'Thursday':4,
                                                          'Friday':5,'Saturday':6,
                                                          'Sunday':7})
df['hour'] = pd.to_datetime(df['Time']).dt.hour
df['minute'] = pd.to_datetime(df['Time']).dt.minute
df['Temp'] = df['Time'].apply(lambda x: x.split(' ')[1])
df['AM/PM'] = df['Temp'].replace({'AM':0,'PM':1})
df= df.drop(columns = ['Time','Temp'], axis=1)
df

print(df.head())

# Select independent and dependent variable
X = df.drop(['Traffic Situation'],axis=1)
y = df["Traffic Situation"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10,shuffle=True)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Instantiate the model
classifier = SVC()
# Fit the model
classifier.fit(X_train, y_train)

# Save the model and the scaler to pickle files
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(sc, open("scaler.pkl", "wb"))
