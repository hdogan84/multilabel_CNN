# (conda install -c anaconda xlrd) # not working
# conda install -c anaconda openpyxl

import os
import pandas as pd
import soundfile as sf
import json

result_dir = "./WS-ARSU2022-pos-neg/"
root_dir = "/mnt/z/Projekte/DeViSe/"

def process_json2excel(model_name):

    class_index = 1  #Crex crex=0  & Waldschnepfe=1

    model_dir = result_dir + model_name + "-1/"

    src_dir = "./data_hkn/"
    input_xls_file = src_dir + "ScolopaxRusticolaAnnotations_v28_5s_Scores.xlsx"
    output_xls_file = src_dir + "ScolopaxRusticolaAnnotations_v29_5s_Scores.xlsx"

    df = pd.read_excel(input_xls_file, engine='openpyxl')
    df_new = df
    df_new[model_name] = None # add new column with the model name

    # !! this part needed when the input is Mario's excel file
    #for i in range(len(df_new)):
    #    df_new.at[i,"filename"]=df.at[i,"filename"][:-4]
    
    print(df_new["filename"])

    # go through results
    count=0
    files = os.listdir(model_dir)

    # check the lengths of testsets
    for file in files:
        f=open(model_dir+file)
        result_file = json.load(f)
        file_id = result_file["fileId"]
        result_arr = result_file["channels"][0]
        count += len(result_arr)
        idx=df.index[df['filename']==file_id].tolist()
        #print(idx)
        #print(len(result_arr))
        
        if len(idx) != len(result_arr): 
            raise Exception("Mismatch with testset lengths for file {}".format(file_id))
 
        for i, result in enumerate(result_arr):
            probability = result['predictions']['probabilities'][class_index]
            df_new.at[idx[i],model_name] = probability
    
    if count != len(df): 
        raise Exception("Mismatch with testset lengths")

    df_new.to_excel(output_xls_file, index = False)

process_json2excel("devise-221117-v1-ep150")

    
print("Done.")
