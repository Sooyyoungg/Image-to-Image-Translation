import pandas as pd
import numpy as np
import csv

t1_t2 = pd.read_csv('structure_qc_list_8922.csv', header=None)
diff = pd.read_csv('dwi_qc_list_1436.csv', header=None)

f = open('common_qc_list.csv', 'w', newline='')
wr = csv.writer(f)

row = 0
for s in np.array(t1_t2.iloc[:,0]):
    if s in np.array(diff.iloc[:,0]):
        wr.writerow([row, s])
        row = row + 1
