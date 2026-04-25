import typer
from typing import Annotated
from back import ColumnStats, OpenFile, CleanData, ColsCorrelation, plot_all, plot_scatter, run_ml
from rich import print
app = typer.Typer()

def separate_columns(columns) -> list:
    cols = [x.strip() for x in columns.split(",")] if columns else print("[bold red]Choose at least 1 column[/bold red]")
    try:
        cols = [col.replace("_", " ") for col in cols]
    except TypeError:
        return "0"
    else: return cols

@app.command()
def main(file: str,
        df_clean: Annotated[bool, typer.Option("--clean/--raw", help="Clean data and log changes")] = True,
        df_stats: Annotated[bool, typer.Option("--stats", help="Show column's stats. Use for one column")] = False,
        df_corr: Annotated[bool, typer.Option("--corr", help="choose two columns")] = False,
        df_plots: Annotated[bool, typer.Option("--plots", help="build plot")] = False,
        df_ml: Annotated[bool, typer.Option("--ml", help="prediction part")] = False,
        
        show: Annotated[bool, typer.Option("-sh", help="show the dataframe")] = False,
        row: Annotated[int, typer.Option("-row", help="number of rows to show")] = 0,
        columns: Annotated[str, typer.Option("-col", help="Choosed columns")] = None,
        z_score: Annotated[float, typer.Option("-zs", help="Score for stats")] = 5.0,
        
        density_plot: Annotated[bool, typer.Option("-dp", help="Tag to show density plot")] = False,
        hist_plot: Annotated[bool, typer.Option("-hp", help="Tag to show hist plot")] = False,
        box_plot: Annotated[bool, typer.Option("-bp", help="Tag to show basic box plot")] = False,
        box_plot_2nd: Annotated[bool, typer.Option("-bpt", help="Tag to show box plot 2nd")] = False,
        scatter_plot: Annotated[bool, typer.Option("-sp", help="Tag to show scatter plot")] = False):


    df = OpenFile(file, row, show)
    
    if df_clean:
        df = CleanData(df, z_score, show)

    if df_stats:
        cols = separate_columns(columns)
        if len(cols) == 1:
            ColumnStats(df, cols)
        else: print ("[bold red]Use only one column for stats[/bold red]")
        
    if df_corr:
        cols = separate_columns(columns)
        if len(cols) == 2:
            ColsCorrelation(df, cols)
        else: print ("[bold red]Use only two columns for correlation[/bold red]")

    if df_plots:
        cols = separate_columns(columns)
        if len(cols) == 1:
            plot_all(df, cols, density_plot, hist_plot, box_plot, box_plot_2nd)
        elif len(cols) == 2 and scatter_plot:
            plot_scatter(df, cols)
        else: print ("[bold red]Choose correct plot parametr[/bold red]")
    
    if df_ml:
        cols = separate_columns(columns)
        if len(cols) == 1:
            run_ml(df, cols)
        else: print ("[bold red]Use only one column for stats[/bold red]")

if __name__ == "__main__":
    app()