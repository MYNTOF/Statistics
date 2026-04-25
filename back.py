import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import contextlib
from rich.errors import NotRenderableError
from rich import print
from rich.table import Table


logging.basicConfig(level=logging.INFO, filename="logroot.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")


def rich_display_dataframe(df, title: str = "Dataframe") -> None: #Table from DataFrame
    
    df = df.astype(str)
    table = Table(title=title)

    for col in df.columns:
        table.add_column(col)
    for row in df.values:
        with contextlib.suppress(NotRenderableError):
            table.add_row(*row)
    print(table)



def select_columns(df, columns: list = None) -> list: #Sub function for selected columns
    
    if len(columns) not in [1, 2]:
        print("[bold red]Choose correct number of columns[/bold red]")
        return 
    for x in columns:
        if x not in df.columns:
            print(f"[bold red] Column wasn't found: {x}[/bold red]")
            return
    return df[columns]



def OpenFile(file: str, stw: int = 0, show: bool = False)-> pd.DataFrame: #Read file
    
    try:
        df = pd.read_csv(file)
        if show:
            rich_display_dataframe(df.head(stw), "[bold green]All existing columns[/bold green]")
        return df
    except Exception as ex:
        print (f"[bold red]Error reading file: {ex}[/bold red]")
        return
    
    

def CleanData(df: pd.DataFrame, z_score: float = 5.0, show: bool = False) -> pd.DataFrame:
    if show:
        rich_display_dataframe(df.head(5),"[bold green]Raw data[bold green]")
    
    df_before = df.shape
    df_clean = df.dropna() #drop NANs
    logging.info(f"{df_before[0] - df_clean.shape[0]} droped rows with NANs")

    df_before = df_clean.shape
    for col in df_clean.select_dtypes(include=["datetime"]): #validate dates
            df_clean[col] = df_clean[col].dt.strftime("%Y-%m-%d")
    logging.info(f"{df_before[0] - df_clean.shape[0]} validated dates")

    df_before = df_clean.shape
    for col in list(df_clean.columns): # Checking columns on numeric and dates, drop others
        numeric = pd.to_numeric(df_clean[col], errors="coerce")

        if numeric.notna().sum() > 0:
            df_clean.loc[:, col] = numeric
            continue

        date = pd.to_datetime(df_clean[col], format="%Y-%m-%d" ,errors="coerce")

        if date.notna().sum() > 0:
            df_clean.loc[:, col] = date
            continue
        df_clean = df_clean.drop(columns=[col])
    logging.info(f"{df_before[1] - df_clean.shape[1]} droped columns without numbers and dates")
    

    
    df_clean = df_clean.dropna()  # Repeat Nan's drop
    logging.info(f"{df_before[0] - df_clean.shape[0]} dropped Nan's rows in rest columns")
    
    df_before = df_clean.shape
    df_clean = df_clean.drop_duplicates() #Duplicates drop
    logging.info(f"{df_before[0] - df_clean.shape[0]} duplicates droped")

    df_before = df_clean.shape
    numeric_cols = df_clean.select_dtypes(include=np.number).columns   #
    for col in numeric_cols:
        if df_clean[col].std() == 0:
            continue
        ZScore = np.abs(stats.zscore(df_clean.loc[:, col]))
        df_clean = df_clean[ZScore < z_score]
    logging.info(f"{df_before[0] - df_clean.shape[0]} z_scores changes")

    if show:
        rich_display_dataframe(df_clean.head(5),"[bold green]Clean data[/bold green]")
        print(f"[bold green]Used z_score = {z_score}[/bold green]")
    return df_clean



def ColumnStats(df, cols: list) -> None:
    cd = select_columns(df, cols)
    if cd is None:
        return
    else:
        cd = cd.iloc[:, 0]
        AllStats = {
        "count": cd.count(),
        "mode": cd.mode().iloc[0],
        "median": cd.median().round(decimals=3),
        "mean": cd.mean().round(decimals=3),
        "min": cd.min(),
        "q1": cd.quantile(0.25),
        "q2": cd.quantile(0.5),
        "q3": cd.quantile(0.75),
        "iqr": cd.quantile(0.75) - cd.quantile(0.25),
        "max": cd.max(),
        "mad_mean": (cd - cd.mean()).abs().mean().round(decimals=3),
        "mad_median": (cd - cd.median()).abs().median().round(decimals=3),
        "std": cd.std().round(decimals=3),
        "variance": cd.var().round(decimals=3),
        "skew": cd.skew().round(decimals=3),
        "kurtosis": cd.kurtosis().round(decimals=3),
        }
        table = Table(title=f"[bold green]Stats of[/bold green] {cols} [bold green]column[/bold green]")
        val = ()
        for k in AllStats.keys():
            table.add_column(k)
            val = val + (str(AllStats[k]),)
        table.add_row(*val)
        print(table)
        return

def ColsCorrelation(df, cols: list) -> None: # Show correlation of 2 columns
    cd = select_columns(df, cols)
    if cd is None:
        return
    else:
        print(f"[bold green]{cd[cols[0]].corr(cd[cols[1]]).round(4)}[/bold green]")
        return
    


def plot_all(df, cols: list, density_plot: bool=False,hist_plot: bool=False, box_plot: bool=False, box_plot_2nd: bool=False) -> None:
    cd = select_columns(df, cols)
    cd = cd.iloc[:, 0]
    if density_plot: # Density Plot
        plt.figure()
        sns.kdeplot(cd, fill=True)
        plt.axvline(cd.mean(), label="mean", color="r")
        plt.axvline(cd.median(), label="median", color="g")
        plt.axvline(cd.mode()[0], label="mode")
        plt.legend()
        plt.show(block=True)
    
    elif hist_plot: # Hist Plot
        plt.figure()
        sns.histplot(cd)
        plt.show(block=True)

    elif box_plot: # Basic Box Plot
        sns.boxplot(cd)
        plt.title("Basic Box Plot")
        plt.show(block=True)
    
    elif box_plot_2nd: # Custom Box Plot
        stats = [{
            'label': 'Custom Box Plot',  # Label for the x-axis
            'q1': cd.mean()-np.std(cd),
            'med': cd.mean(),
            'q3': cd.mean()+np.std(cd),
            'whislo': cd.min(),
            'whishi': cd.max(),
            }]
                                
        fig, ax = plt.subplots()
        ax.bxp(stats,showfliers=False)
        plt.show(block=True)
    
    else: print("Choose plot to render")


def plot_scatter(df, cols: list) -> None:
    cd = select_columns(df, cols)
    plt.figure()
    sns.scatterplot(x=cd[cols[0]],y=cd[cols[1]])
    plt.show(block=True)



def run_ml(df, target) -> None:

    for col in df.select_dtypes(include=["datetime"]):
        df[col] = df[col].map(lambda x: x.timestamp())

    X = df.drop(columns=target)
    y = df[target].iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Decision Tree": DecisionTreeRegressor()
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results[name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

        plt.figure()
        plt.scatter(y_test, pred, alpha=0.6)
        y_test = np.array(y_test).flatten()
        pred = np.array(pred).flatten()
        plt.scatter(y_test, pred, alpha=0.6)
        min_val = min(y_test.min(), pred.min())
        max_val = max(y_test.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.xlabel("Real values")
        plt.ylabel("Predicted values")
        plt.title(f"{name} Prediction")
        plt.show(block=True)

    for model, metrics in results.items():
        print(f"\n[bold green]{model}[/bold green]")
        for k, v in metrics.items():
            print(f"{k}: {round(v, 4)}")

    return
