import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

from Project.Data_PreProcess import Preprocess
from Project.Log_Creation import Logger

# Load the file
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Obj creation for custom class
log = Logger()
data_preprocess = Preprocess()

# File Info
prt_head = f"File Head Data : {data.head()}"
prt_shape = f'Shape of the file : {data.shape}'
prt_dimension = f"File Dimension is : {data.ndim}"
prt_columns_list = f'List of columns in the file : \n{data.columns}'

# log creation
log.create_log(prt_head)
log.create_log(prt_shape)
log.create_log(prt_dimension)
log.create_log(prt_columns_list)

list_of_NaN_Clm = f'List of NaN columns are :\n{data_preprocess.find_nan_clms(data)}'
log.create_log(list_of_NaN_Clm)

# filled NaN in dataset
filled_nan_data = data_preprocess.fill_nan(data)
data = filled_nan_data

# removed NaN in dataset
# data = data_preprocess.drop_nan_rows(data)


'''-------------Plotting------------------'''


def add_plt_txt(plt_obj):
    # Adding value text at the top of each bar
    for p in plt_obj.patches:
        plt_obj.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                         textcoords='offset points')


# Calculate total sales by region
total_sales_by_region = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
prt_total_sales_by_region = f'total_sales_by_region : \n{total_sales_by_region}'
log.create_log(prt_total_sales_by_region)

# Plot the total sales by region
plt.figure(figsize=(10, 6))
ax1 = total_sales_by_region.plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
add_plt_txt(ax1)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales (in millions)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Identify the top-selling game for each platform
top_selling_games_by_platform = data.loc[data.groupby('Platform')['Global_Sales'].idxmax()]

# Select relevant columns for display
top_selling_games_by_platform = top_selling_games_by_platform[['Platform', 'Name', 'Global_Sales']].sort_values(
    by='Global_Sales', ascending=False)

log.create_log(top_selling_games_by_platform)

# Plot the top_selling_games_by_platform
plt.figure(figsize=(13, 7))
ax = sns.barplot(x='Platform', y='Global_Sales', data=top_selling_games_by_platform)
add_plt_txt(ax)
plt.title('Top Selling Games by Platform')
plt.ylabel('Total Sales (in millions)')
plt.xticks(rotation=60)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Identify the top 20 selling games names
top_selling_games_names = data.loc[data.groupby('Name')['Global_Sales'].idxmax()].sort_values(
    by='Global_Sales', ascending=False)
top_selling_games_names = top_selling_games_names.head(20)
log.create_log(top_selling_games_names)

# Plot the top_selling_games_by_platform
plt.figure(figsize=(13, 7), dpi=80)
sns.barplot(y='Name', x='Global_Sales', data=top_selling_games_names, color='firebrick')
plt.title('Top 20 Selling Game Names')
plt.ylabel('Total Sales (in millions)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Count the number of game releases per year
game_releases_per_year = data['Year_of_Release'].value_counts().sort_index()
log.create_log(game_releases_per_year)

# Plot the number of game releases per year
plt.figure(figsize=(13, 7))
game_releases_per_year.plot(kind='line', marker='o')
plt.title('Number of Game Releases Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Releases')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Correlation of numeric columns
numeric_clm = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales',
                    'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']]

plt.figure(figsize=(12, 10), dpi=80)
sns.heatmap(numeric_clm.corr(), xticklabels=numeric_clm.corr().columns,
            yticklabels=numeric_clm.corr().columns,
            cmap='RdYlGn', annot=True)

# Decorations
plt.title('Correlation of numeric columns', fontsize=22)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
log.create_log(numeric_clm.corr())

# Identify the sales_over_genre
sales_over_genre = data.groupby('Genre')['Global_Sales'].sum().reset_index().sort_values(
    by='Global_Sales', ascending=False)
log.create_log(sales_over_genre)

# plot top_selling_genre
plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Genre', y='Global_Sales', data=sales_over_genre)
add_plt_txt(ax)
plt.title('Sales analyse by Genres')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Identify the sales_over_platform
sales_over_platform = data.groupby('Platform')['Global_Sales'].sum().reset_index().sort_values(
    by='Global_Sales', ascending=False)
log.create_log(sales_over_platform)

# plot sales_over_platform
plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Platform', y='Global_Sales', data=sales_over_platform)
add_plt_txt(ax)
plt.title('Sales analyse by Platform')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Aggregate sales by year
sales_by_year = data.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
log.create_log(sales_by_year)

# Convert Year_of_Release to int
sales_by_year['Year_of_Release'] = sales_by_year['Year_of_Release'].astype(int)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(sales_by_year['Year_of_Release'], sales_by_year['Global_Sales'], marker='o')
plt.title('Global Sales Over the Years')
plt.xlabel('Year of Release')
plt.ylabel('Global Sales (in millions)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


'''----------------Different Models------------------------'''
# Select features and target variable for model
features = ['Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales',
            'Other_Sales', 'Critic_Score', 'Critic_Count', 'User_Score',
            'User_Count', 'Rating']
target = 'Global_Sales'

X = data[features]
y = data[target]

# Encode categorical variables
numerical_features = ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
                      'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
categorical_features = ['Platform', 'Genre', 'Publisher', 'Rating']

# apply scaler and encoding for train
preprocessor = ColumnTransformer(
    transformers=[('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_features),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                  ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and test data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


def plt_and_evaluate_model(y_model_predict, model_name):
    # Plot the model predictions
    plt.scatter(y_test, y_model_predict, alpha=0.3, label=model_name)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r')
    plt.xlabel('Actual Global Sales (y_test)')
    plt.ylabel('Predicted Global Sales (y_predict)')
    plt.title('Actual vs Predicted Global Sales')
    plt.legend()
    plt.show()
    # Evaluate the model
    mse = f'Mean Squared Error of {model_name} : {mean_squared_error(y_test, y_model_predict)}'
    r2 = f'R Squared Score value of {model_name} : {r2_score(y_test, y_model_predict)}'
    log.create_log(mse)
    log.create_log(r2)


# Train a linear regression model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# Make predictions
y_predict = model_linear.predict(X_test)
mse = f'Mean Squared Error of Linear regression : {mean_squared_error(y_test, y_predict)}'
r2 = f'R Squared Score value of Linear regression : {r2_score(y_test, y_predict)}'
log.create_log(mse)
log.create_log(r2)

# Plot the results
plt.figure(figsize=(12, 7))

# Scatter plot of actual vs predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_predict, alpha=0.5, color='blue', label='Linear Regression')
plt.plot([0, max(y_test)], [0, max(y_test)], 'r')
plt.xlabel('Actual Global Sales (y_test)')
plt.ylabel('Predicted Global Sales (y_predict)')
plt.title('Actual vs Predicted Global Sales')
plt.legend()

# Residual plot
plt.subplot(1, 2, 2)
sns.residplot(x=y_test, y=y_predict, lowess=True)
plt.xlabel('Actual Global Sales')
plt.ylabel('Residuals')
plt.title('Residuals of Predicted Global Sales')
plt.tight_layout()
plt.show()

# Create and Train the random forest regressor model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate the model
y_predict_rf = model_rf.predict(X_test)
plt_and_evaluate_model(y_predict_rf, 'RandomForestRegressor Model')

'''-------------Check Outliers------------------'''


def plot_outliers(columns, plot_title):
    plt.figure(figsize=(10, 6), dpi=80)
    sns.boxplot(data=columns)
    plt.xticks(rotation=50)
    plt.title(plot_title)
    plt.show()


# Plot Outliers
plot_outliers(data, 'Checking Outliers')
prt_shape = f'Before removing outliers (shape) : {data.shape}'
log.create_log(prt_shape)

# Remove outliers from the numerical columns
df_no_outliers = data_preprocess.remove_outliers(data, numerical_features)
plot_outliers(df_no_outliers, 'Removed the Outliers')

prt_shape = f'After removing outliers (shape) : {df_no_outliers.shape}'
log.create_log(prt_shape)

'''---------------DecisionTree & KNeighbors-----------------'''

# Create and Train a Decision Tree model
model_tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1, min_samples_split=10)
model_tree.fit(X_train, y_train)

# Make predictions
y_predict_dt = model_tree.predict(X_test)
plt_and_evaluate_model(y_predict_dt, 'Decision Tree Model')


# Create and train the model
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(X_train, y_train)

# Make predictions
y_predict_knr = knn_model.predict(X_test)
plt_and_evaluate_model(y_predict_knr, 'KNeighborsRegressor Model')


log.create_log('--------------------Program Finished-----------------------')
