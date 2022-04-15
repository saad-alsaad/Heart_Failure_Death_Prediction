import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, matthews_corrcoef, \
    mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split


def linear_reg(x: pd.DataFrame, y: pd.DataFrame):
    linear_model = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
    linear_model.fit(X_train, Y_train)
    Y_pred = linear_model.predict(X_test)
    r_sq = linear_model.score(X_train, Y_train)
    print('coefficient:', r_sq)
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, Y_pred)))


def ploy_reg(x: pd.DataFrame, y: pd.DataFrame):
    poly_reg = PolynomialFeatures(degree=1)
    X_poly = poly_reg.fit_transform(x)
    lin_reg2 = LinearRegression()
    model = lin_reg2.fit(X_poly, y)

    # Store our predicted Humidity values in the variable y_new
    y_pred = lin_reg2.predict(X_poly)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print("Polynomial coff: ", model.score(X_poly, y))
    print("RMSE: ", rmse)
    print("R2: ", r2)
    # Plot our model on our data
    print("Mean squared error: ", mean_squared_error(y, y_pred))
    print("Mean absolute error: ", mean_absolute_error(y, y_pred))
    print("Mean absolute percentage error: ", mean_absolute_percentage_error(y, y_pred))
    plt.scatter(x, y, c="black")

    plt.xlabel(x.columns.values[0])
    plt.ylabel(y.columns.values[0])
    plt.plot(x, y_pred)
    plt.show()
    plt.clf()

    # predict a value
    y_new = lin_reg2.predict(poly_reg.fit_transform([[0.6]]))
    print("y_new: ", y_new)
    plt.scatter(0.6, y_new, c="red")
    plt.xlabel("serum_creatinine")
    plt.ylabel("age")
    plt.plot(x, y_pred)
    plt.show()
    plt.clf()


def show_boxplot(df_data: pd.DataFrame):
    df_data[['age', 'time', 'ejection_fraction', 'platelets', 'creatinine_phosphokinase']]\
        .boxplot(grid=True, figsize=(10, 10))
    plt.show()
    plt.clf()
    df_data[['serum_sodium', 'serum_creatinine']].boxplot(grid=True)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    df_data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    #sns.scatterplot(x='sepal_length', y ='petal_length' ,
    #data = df , hue = 'species')

    first_quantile = df_data.quantile(0.25).to_frame()
    second_quantile = df_data.quantile(0.5).to_frame()
    third_quantile = df_data.quantile(0.75).to_frame()

    first_quantile.to_csv("output/files/first_quantile.csv", index_label=['feature'], header=['result'])
    second_quantile.to_csv("output/files/second_quantile.csv", index_label=['feature'], header=['result'])
    third_quantile.to_csv("output/files/third_quantile.csv", index_label=['feature'], header=['result'])

    # df_data['DEATH_EVENT'].replace(0, 'NO', inplace=True)
    # df_data['DEATH_EVENT'].replace(1, 'YES', inplace=True)

    df_data['platelets'] = df_data['platelets'].apply(lambda x: x/10000)
    df_data['creatinine_phosphokinase'] = df_data['creatinine_phosphokinase'].apply(lambda x: x/100)
    df_data['serum_sodium'] = df_data['serum_sodium'].apply(lambda x: x/10)
    df_data['time'] = df_data['time'].apply(lambda x: x/10)
    cleaned_df = df_data[(df_data['serum_creatinine'] <= 1.7) & (df_data['ejection_fraction'] <= 67.5)]
    cleaned_df.to_csv('cleaned_output.csv', index=False) # to be used in Weka
    show_boxplot(df_data)

    y = cleaned_df[['age']]
    x = cleaned_df[['serum_creatinine']]

    # ploy_reg(x, y)
    # linear_reg(x,y)

    independent_variable = df_data.drop('DEATH_EVENT', axis=1)

    sns.kdeplot(x='age', data=df_data[['DEATH_EVENT', 'age']], y='DEATH_EVENT')
    plt.savefig('output/plots/age_DEATH_EVENT_density_plot.png')
    plt.clf()

    sns.displot(df_data[['DEATH_EVENT', 'age']], x='age', y='DEATH_EVENT', kind="kde")
    plt.savefig('output/plots/age_DEATH_EVENT_displot.png')
    plt.clf()

    sns.displot(df_data[['DEATH_EVENT', 'age']], x='age', kind="kde", hue="DEATH_EVENT", multiple="stack")
    plt.savefig('output/plots/age_DEATH_EVENT_displot2.png')
    plt.clf()

    a4_dims = (22, 50)
    fig, ax = plt.subplots(figsize=a4_dims)
    df_data.plot(kind='density', subplots=True, ax=ax)
    plt.savefig('output/plots/density_plot.png')
    plt.clf()

    independent_variable_corr = independent_variable.corr()
    a4_dims = (22, 15)
    fig, ax = plt.subplots(figsize=a4_dims)
    res = sns.scatterplot(ax=ax, data=independent_variable_corr)
    plt.savefig('output/plots/independent_variables_corr.png')
    plt.clf()

    fig, ax = plt.subplots(figsize=a4_dims)
    res = sns.barplot(ax=ax, data=independent_variable_corr)
    plt.savefig('output/plots/bar_independent_variables_corr.png')
    plt.clf()

    sns.heatmap(independent_variable_corr, annot=True, fmt="10.2f", cmap="YlGnBu")
    plt.savefig('output/plots/independent_variables_corr_heatmap.png')
    plt.clf()

    sns.heatmap(df_data.corr(), annot=True, fmt="10.2f", cmap="YlGnBu")
    plt.savefig('output/plots/df_data_corr_heatmap.png')
    plt.clf()

    a4_dims = (22, 15)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.scatterplot(ax=ax, data=df_data)
    plt.savefig('output/plots/variables_corr.png')
    plt.clf()

    a4_dims = (14, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.scatterplot(ax=ax, data=df_data[['DEATH_EVENT', 'age']].corr(), x='age', y='DEATH_EVENT')
    plt.savefig('output/plots/DEATH_EVENT_age_correlation_plot.png')
    plt.clf()

    a4_dims = (14, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.scatterplot(ax=ax, data=df_data[['serum_creatinine', 'age']], x='serum_creatinine', y='age')
    plt.savefig('output/plots/serum_creatinine_age_scatter_plot.png')
    plt.clf()

    a4_dims = (14, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.scatterplot(ax=ax, data=df_data[['DEATH_EVENT', 'serum_creatinine']].corr(), x='serum_creatinine', y='DEATH_EVENT')
    plt.savefig('output/plots/DEATH_EVENT_serum_creatinine_correlation_plot.png')
    plt.clf()

    a4_dims = (14, 10)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.scatterplot(ax=ax, data=df_data[['DEATH_EVENT', 'age']], x='age', y='DEATH_EVENT')
    plt.savefig('output/plots/DEATH_EVENT_age_scatter_plot.png')
    plt.clf()

    sns.kdeplot(x='DEATH_EVENT', y='age', data=df_data, color='black')
    plt.savefig('output/plots/DEATH_EVENT_age_density_plot.png')
    plt.clf()

    sns.kdeplot(x='DEATH_EVENT', data=df_data, color='black')
    plt.savefig('output/plots/DEATH_EVENT_plot.png')
    plt.clf()
