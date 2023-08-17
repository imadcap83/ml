import pandas as pd
from pycaret.classification import *

df = pd.read_csv('sourcedata.csv')

setup_target = setup(data=df, target='Transported')

best_model = compare_models()

results = pull()

type(results)

