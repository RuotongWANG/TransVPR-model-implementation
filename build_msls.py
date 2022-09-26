# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:03:26 2021

@author: 24379
"""

from os.path import join
import numpy as np
import json
import pandas as pd

default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'toy' :["london"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}

def build_subtask(part):
    citys = default_cities[part]
    subdir = 'test'
    qPaths = []
    dbPaths = []
    for city in citys:
        print('City: ' + city)
        qIdx = pd.read_csv(join(subdir, city, 
                                'query', 'subtask_index.csv'), index_col = 0)
        dbIdx = pd.read_csv(join(subdir, city, 
                                 'database', 'subtask_index.csv'), index_col = 0)
        qDataRaw = pd.read_csv(join(subdir, city, 
                                    'query', 'raw.csv'), index_col = 0)
        dbDataRaw = pd.read_csv(join(subdir, city, 
                                     'database', 'raw.csv'), index_col = 0)
        
        # filter of pano
        n_pano_q = np.where((qDataRaw['pano'] == False).values)[0]
        n_pano_db = np.where((dbDataRaw['pano'] == False).values)[0]
        # filter of subtask
        sub_q = np.where(qIdx['all'])[0]
        sub_db = np.where(dbIdx['all'])[0]
        
        val_q = np.intersect1d(n_pano_q, sub_q)
        val_db = np.intersect1d(n_pano_db, sub_db)
        q_keys = qIdx['key'].values[val_q]
        db_keys = dbIdx['key'].values[val_db]
        
        qPath = [join(subdir, city, 'query', 'images', key + '.jpg') \
                 for key in q_keys]
        dbPath = [join(subdir, city, 'database', 'images', key + '.jpg') \
                 for key in db_keys]
        qPaths += qPath
        dbPaths += dbPath
    
    print('N query: ', len(qPaths))
    print('N database: ', len(dbPaths))
    with open(part+'_q_sub_all.json',"w") as f:
        json.dump({'qPath': qPaths}, f)   
    with open(part+'_db_sub_all.json',"w") as f:
        json.dump({'dbPath': dbPaths}, f)   
        



if __name__ == "__main__":
    build_subtask('train')
    build_subtask('val')
        
        