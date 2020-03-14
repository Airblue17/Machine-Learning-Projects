# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:23:39 2019

@author: nitin
"""

import csv

def write_csv(predictions, file_path):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Id", "Predicted"])

        index = 1

        for pred in predictions:
            writer.writerow([index, pred])
            index += 1