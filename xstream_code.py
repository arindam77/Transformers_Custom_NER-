#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing Simpletransformers

pip install simpletransformers


# In[ ]:




#Connecting to Google Drive for extracting tar.gz file content

!tar -xzvf "/content/drive/MyDrive/xstream.tar.gz" -C "/content/drive/MyDrive"


# In[ ]:


#Loading the training pickle file

import pickle


with open('/content/drive/MyDrive/xstream-nlp-task/train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    


# In[ ]:


##Training Data Preparation for ner_modelling


#Converting list of tuples to list of lists

list_elem = [list(ele) for ele in train_data]  


#Creating Training Dataframe for Training Transformer ner_model


import pandas as pd

df = pd.DataFrame(list_elem)

df.columns = ['words', 'tags']


words = df['words'].to_list()
tags = df['tags'].to_list()



df_tr_data = pd.DataFrame(zip(words, tags), columns=['words', 'tags'])
df_tr_data['index_col'] = df_tr_data.index

df_tr_data['index_col'] = 'Sentence' + df_tr_data['index_col'].astype(str)



#Combining Sentenceindex, text and tags to prepare the dataframe

newdf = []
for _, line in df_tr_data.iterrows():
        for txt, tag in zip(line["words"], line["tags"]):
            newdf.append({'sentence_id': line["index_col"], 'words': txt, 'labels':tag})
            
df_train= pd.DataFrame(newdf)

#Checking structure of the training dataframe 

df_train.head()


# In[ ]:


#Extracting all distinct labels from the training dataframe

label = df_train["labels"].unique().tolist()            
            
#print(label)    


# In[ ]:


#Using LabelEncoder to encode the sentence_id

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train['sentence_id'] = le.fit_transform(df_train['sentence_id'])
#print(df_train)


# In[ ]:


##Test data preparation for prediction

#Loading the test pickle file


with open('/content/drive/MyDrive/xstream-nlp-task/test.pkl', 'rb') as f:
    test_data = pickle.load(f)   

    
#Converting list of tuples to list of lists
    

test_elem = [list(ele) for ele in test_data]  



#Creating Test Dataframe for prediction


df = pd.DataFrame(test_elem)
df.columns = ['words', 'tags']


words = df['words'].to_list()
tags = df['tags'].to_list()



df_ts_data = pd.DataFrame(zip(words, tags), columns=['words', 'tags'])
df_ts_data['index_col'] = df_ts_data.index

df_ts_data['index_col'] = 'Sentence' + df_ts_data['index_col'].astype(str)


#Combining Sentenceindex, text and tags to prepare the Test dataframe


testdf = []
for _, line in df_ts_data.iterrows():
        for txt, tag in zip(line["words"], line["tags"]):
            testdf.append({'sentence_id': line["index_col"], 'words': txt, 'labels':tag})
            
            
df_test_= pd.DataFrame(testdf)

#Encoding the test sentence_id


df_test_['sentence_id'] = le.fit_transform(df_test_['sentence_id'])



# In[ ]:


#Importing the required libraries

from simpletransformers.ner import NERner_model, NERArgs



Initialising NERner_model for training


##Experiments with different ner_models & parameters

ner_model = NERner_model(
   "roberta", "roberta-base", labels = label, args = {"output_dir": "xstream_ner_model"}



args={"save_eval_checkpoints": False,
      "save_steps": -1,
      "output_dir": "xstream_ner_model",
      'overwrite_output_dir': True,
     "save_ner_model_every_epoch": False,
     'reprocess_input_data': True, 
     "train_batch_size": 10,'num_train_epochs': 5,"max_seq_length": 256, "gradient_accumulation_steps": 8})

ner_model = NERner_model(
   "roberta", "roberta-base", labels = label, args = args




ner_model = NERner_model(
    "roberta", "roberta-base", labels = label, args = {"output_dir": "xstream_ner_model"}





ner_model = NERner_model(
    "roberta", "xstream_ner_model/checkpoint-2000", labels = label, args = {"output_dir": "xstream_model", 'overwrite_output_dir': True}



ner_model = NERner_model('bert', 'bert-base-cased'
  , labels = label, args = {"output_dir": "xstream_ner_model_Final"}
)

#Training Bert ner_model

ner_model.train_ner_model(df_train)


# In[ ]:


#Evaluating ner_model Performance

result, ner_model_outputs, predictions = ner_model.eval_ner_model(df_test_)
print(result)
print(ner_model_outputs)

