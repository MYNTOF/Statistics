
## Quick Start
### Run from main folder
```bash
C:\...\Statistics\python proj.py data\file.csv ...
```
### All testing data stored in ```\Data```

- 2019.csv
- health.csv
- train.csv

## Main options
```python
--clean  #always included but can be ignored by using --raw
    -zs # use to write your own z score for outliers detection (default value = 5.0)
    -sh # will show basic vision of data befor and after function

--stats #shows basic stats for 1 column from database (-col always required)
    -col #requires 1 parametr "["Column_Name"]" with _ instead of " "

--corr #shows correlation between 2 columns from database (-col always required)
    -col #requires 2 parametrs "Column_Name,Column_Name" with _ instead of " "

--plots #renders choosed plot for 1 or 2 columns (-col always required)
    -dp #renders density plot for 1 column
    -hp #renders hist plot for 1 column
    -bp #renders box plot for 1 column
    -bpt #renders box plot for 1 column
    -sp #renders scatter plot for 2 columns

--ml #Predict future values for chosen column (-col always required)
```
## Function examples
### Show data
```bash
C:\example\Statistics\python proj.py data\file.csv -sh

  All existing columns
┌──────────┬──────────┐
│ Column A │ Column B │ 
├──────────┼──────────┤
└──────────┴──────────┘
        Raw data
┌──────────┬──────────┐
│ Column A │ Column B │ 
├──────────┼──────────┤
│ Value A  │   Nan    │
├──────────┼──────────┤
│   ....   │   ....   │
├──────────┼──────────┤
│ Value C  │  Value D │
└──────────┴──────────┘
        Clean data
┌──────────┬──────────┐
│ Column A │ Column B │ 
├──────────┼──────────┤
│   ....   │   ....   │
├──────────┼──────────┤
│ Value C  │  Value D │
└──────────┴──────────┘
```
All changes rewrites in ```logroot.log```
### Show statisticks
```bash
C:\example\Statistics\python proj.py data\file.csv --stats -col Ship_Number
                                                       
                                                        Stats of ['Ship_Number'] column
┌───────┬──────┬────────┬────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────────┬────────────┬────────┬──────────┬────────┬──────────┐
│ count │ mode │ median │ mean   │ min  │ q1   │ q2   │ q3   │ iqr  │ max  │ mad_mean │ mad_median │ std    │ variance │ skew   │ kurtosis │
├───────┼──────┼────────┼────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────────┼────────────┼────────┼──────────┼────────┼──────────┤
│ 131   │ 19.0 │ 36.0   │ 35.675 │ 0.92 │ 24.5 │ 36.0 │ 48.0 │ 23.5 │ 80.0 │ 13.335   │ 12.0       │ 16.391 │ 268.658  │ -0.085 │ -0.38    │
└───────┴──────┴────────┴────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────────┴────────────┴────────┴──────────┴────────┴──────────┘
```
### Correlation function
```bash
C:\example\Statistics\python proj.py data\file.csv --stats -col Ship_Number,DockID
                                                       
-0.1334
```
### Density plot
```bash
C:\example\Statistics\python proj.py data\file.csv --plots -col Fare -dp

```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/dp_show.jpg)
### Hist plot (basic)
```bash
C:\example\Statistics\python proj.py data\file.csv --plots -col Fare -hp

```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/hp_show.jpg)
### Box plot (basic)
```bash
C:\example\Statistics\python proj.py data\file.csv --plots -col Fare -bp

```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/bp_show.jpg)
### Box plot (modified)
```bash
C:\example\Statistics\python proj.py data\file.csv --plots -col Fare -bpt

```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/bpt_show.jpg)
### Scatter plot
```bash
C:\example\Statistics\python proj.py data\file.csv --plots -col Fare,Age -sp

```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/sp_show.jpg)
### ML prediction function
```bash
C:\example\Statistics\python proj.py data\file.csv --ml -col Score

Linear Regression
MSE: 0.0411
MAE: 0.1596
R2: 0.9724

Random Forest
MSE: 0.0104
MAE: 0.0511
R2: 0.993

Decision Tree
MSE: 0.0094
MAE: 0.0597
R2: 0.9937
```
![Image alt](https://github.com/MYNTOF/Statistics/raw/main/img/pred_pt.jpg)
