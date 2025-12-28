import pandas as pd
import os
import sys
from pathlib import Path

#get paths
current_dir= Path(__file__).parent.absolute()
project_root= current_dir.parent
data_dir= project_root/"data"


def load_nd_clean(data_name):

    # load data based on file extension
    file_ext= os.path.splitext(data_name)[1].lower()

    # data path to be loaded based on file extension
    data_path= data_dir/data_name
    if file_ext =='.csv':
        df= pd.read_csv(data_path)
    elif file_ext in ['.xlsx','.xls']:
        df= pd.read_excel(data_path)
    elif file_ext in['.parquet','.pq']:
        df= pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported File format: {file_ext}, use CSV, Excel, or Parquet")
    
    #Data cleaning
    prev_cols= list(df.columns)
    dropped_cols=[]

    #checking for and dropping all NAN columns
    nan_cols= df.columns[df.isna().all()].tolist()
    if nan_cols:
        df= df.drop(columns=nan_cols)
        dropped_cols.extend(nan_cols)
        print(f"Dropped {len(nan_cols)} entirely Nan columns: {nan_cols}")
    
    #returning df
    return df


print(load_nd_clean('Birth.csv'))
    
    