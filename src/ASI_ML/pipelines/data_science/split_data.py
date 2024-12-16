from sklearn.model_selection import train_test_split

def split_data(data, parameters):

    print("Parameters received split:", parameters)

    X = data[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]
    Y = data[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    X_dev, X_test, Y_dev, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=parameters["random_state"]
    )

    path = "gs://solar_defender_data/"

    X_train.to_csv(f"{path}X_train.csv", index=False)
    Y_train.to_csv(f"{path}Y_train.csv", index=False)
    X_dev.to_csv(f"{path}X_dev.csv", index=False)
    Y_dev.to_csv(f"{path}Y_dev.csv", index=False)
    X_test.to_csv(f"{path}X_test.csv", index=False)
    Y_test.to_csv(f"{path}Y_test.csv", index=False)

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test
