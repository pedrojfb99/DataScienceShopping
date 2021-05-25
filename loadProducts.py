import pandas as pd


if __name__ == '__main__':


    df = pd.read_csv("ola.csv")
    print(df.columns)
    #df.to_csv("ola.csv")