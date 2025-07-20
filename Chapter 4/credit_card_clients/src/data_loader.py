import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from path import Path

def load_credit_default_dataset(path: Path = Path("data/default_of_credit_card_clients.csv")):
    df = pd.read_csv(path, header=1, sep=";")

    for col in df.columns:
        if "default" in col.lower() and "payment" in col.lower():
            df.rename(columns={col: "target"}, inplace=True)
            break

    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)


    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test