class Preprocess:
    def find_nan_clms(self, data):
        # Finding which are columns has NaN
        list_of_NaN_Clm = []
        columns = data.columns
        for key in columns:
            if data[key].isna().sum() > 0:
                list_of_NaN_Clm.append(data[key].name)
        return list_of_NaN_Clm

    def drop_nan_rows(self, data):
        data.dropna(axis=0, how="any", inplace=True)
        return data

    def fill_nan(self, data):
        # Fill missing values in numerical columns with the median
        data['Year_of_Release'] = data['Year_of_Release'].fillna(data['Year_of_Release'].median())
        data['Critic_Score'] = data['Critic_Score'].fillna(data['Critic_Score'].median())
        data['Critic_Count'] = data['Critic_Count'].fillna(data['Critic_Count'].median())
        data['User_Score'] = data['User_Score'].fillna(data['User_Score'].median())
        data['User_Count'] = data['User_Count'].fillna(data['User_Count'].median())

        # Fill missing values in categorical columns with the mode
        data['Publisher'] = data['Publisher'].fillna(data['Publisher'].mode()[0])
        data['Developer'] = data['Developer'].fillna(data['Developer'].mode()[0])
        data['Rating'] = data['Rating'].fillna(data['Rating'].mode()[0])
        return data

    def remove_outliers(self, df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
