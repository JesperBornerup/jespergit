import pandas as pd
import mlflow
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
#import sys

polyFeatures = 3 #int(sys.argv[1]) if len(sys.argv) > 1 else 3
estimators =  100 #int(sys.argv[2]) if len(sys.argv) > 1 else 100

df = pd.read_json("dataset.json", orient="split")
gen_df = pd.DataFrame(df['Total']).resample('3h').mean()
wind_df = pd.DataFrame(df[['Direction','Speed']])
df = pd.concat([wind_df, gen_df], axis=1)
df = df.dropna()
#mlflow.set_tracking_uri("http://training.itu.dk:5000")
#mlflow.set_experiment("jesperTry")	


with mlflow.start_run(run_name="<JesperForest>"):
    
    ohe = OneHotEncoder(sparse=False)
    poly = PolynomialFeatures(degree=polyFeatures, include_bias=False)

    trans = ColumnTransformer([
     ("OneHotEncoder", ohe, ["Direction"]),
    ('poly',poly, ['Speed']),
    ], remainder="passthrough")
    
    forrest = RandomForestRegressor(n_estimators=estimators)

    pipeline = Pipeline([
    ('trans', trans),
    ('std_scaler', StandardScaler()),
    ('forrest',forrest)
    ])

    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = pipeline.fit(x_train,y_train)
    predictions = pipeline.predict(x_test)
    truth = y_test
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("polyFeatures", polyFeatures)
    mlflow.log_param("estimators", estimators)
    mlflow.sklearn.save_model(model, "model")

    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
        score = func(truth, predictions)
        print(f"Score {name}: {score}")
        mlflow.log_metric(f"mean_{name}", score)
        

            



    
