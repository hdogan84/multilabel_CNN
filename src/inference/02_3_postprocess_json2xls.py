# (conda install -c anaconda xlrd) # not working
# conda install -c anaconda openpyxl

import os
import pandas as pd
import soundfile as sf
import json

result_dir = "./WS-ARSU2022-pos-neg/"
root_dir = "/mnt/z/Projekte/DeViSe/"

def process_json2excel(model_name,class_index):

    # class_index: Crex crex=0; Waldschnepfe=1

    model_dir = result_dir + model_name + "-1/"

    src_dir = "./data_hkn/"
    input_xls_file = src_dir + "ScolopaxRusticolaAnnotations_v26_5s_Scores.xlsx"
    output_xls_file = src_dir + "ScolopaxRusticolaAnnotations_v30_5s_Scores.xlsx"

    #input_xls_file = src_dir + "CrexCrexAnnotations_v20_5s_Scores.xlsx"
    #output_xls_file = src_dir + "CrexCrexAnnotations_v21_5s_Scores.xlsx"

    df = pd.read_excel(input_xls_file, engine='openpyxl')
    df_new = df
    df_new[model_name] = None # add new column with the model name

    if "file_id" not in df_new.columns: 
        df_new["file_id"] = None
        for i in range(len(df_new)):
            df_new.at[i,"file_id"]=df.at[i,"filename"][:-4]
    
    print(df_new["file_id"])

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
        idx=df_new.index[df_new['file_id']==file_id].tolist()
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

process_json2excel("ammod-220412-v10-ep91",17)

print("Done.")
