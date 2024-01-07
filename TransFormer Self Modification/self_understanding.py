from sklearn.preprocessing import StandardScaler

# Apply Standardization
scaler = StandardScaler()
train_data[train_columns] = scaler.fit_transform(train_data[train_columns])
test_data[test_columns] = scaler.transform(test_data[test_columns])

from sklearn.feature_selection import SelectKBest, f_regression

# Select top k features based on F-statistic
k = 10
selector = SelectKBest(score_func=f_regression, k=k)
selected_features = selector.fit_transform(train_data[train_columns], y_train)

from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
train_data_poly = poly.fit_transform(train_data[train_columns])
test_data_poly = poly.transform(test_data[test_columns])


train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek
train_data['hour_of_day'] = train_data['timestamp'].dt.hour
# Similarly for test data

train_data['feature1_times_feature2'] = train_data['feature1'] * train_data['feature2']

window = 3  # Adjust the window size
train_data['rolling_mean_feature1'] = train_data['feature1'].rolling(window=window).mean()

from sklearn.decomposition import PCA

pca = PCA(n_components=5)  # Adjust the number of components
reduced_features = pca.fit_transform(train_data[train_columns])

import category_encoders as ce

encoder = ce.TargetEncoder()
train_data['encoded_category'] = encoder.fit_transform(train_data['category'], y_train)

train_data['feature1_times_feature2'] = train_data['feature1'] * train_data['feature2']




