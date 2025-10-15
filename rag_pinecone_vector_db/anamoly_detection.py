# Lesson 6 - Anomaly Detection
# Import the Needed Packages
import warnings
warnings.filterwarnings('ignore')
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from DLAIUtils import Utils
import torch
import time
import torch
import os

# Setup Pinecone
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

pinecone = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)
pinecone.create_index(name=INDEX_NAME, dimension=256, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-west-2'))
index = pinecone.Index(INDEX_NAME)

# Load the Dataset
# Note: To access the dataset outside of this course, just copy the following three lines of code and run it (remember to uncomment them first before executing):

#!wget -q --show-progress -O training.tar.zip "https://www.dropbox.com/scl/fi/rihfngx4ju5pzjzjj7u9z/lesson6.tar.zip?rlkey=rct9a9bo8euqgshrk8wiq2orh&dl=1"

#!tar -xzvf training.tar.zip

#!tar -xzvf lesson6.tar

# !head -5 sample.log
# !head -5 training.txt
# Check cuda and Setup the Model
# We are using bert-base-uncased sentence-transformers model that maps sentences to a 256 dimensional dense vector space.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=768)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=device)
device

# Train the Model
train_examples = []
with open('./training.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            a, b, label = line.split('^')
            train_examples.append(InputExample(texts=[a, b], label=float(label)))

#Define dataset, the dataloader and the training loss
warmup_steps=100
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# (Note: load_pretrained_model = True): We've saved the trained model and are loading it here for speedier results, allowing you to observe the outcomes faster. Once you've done an initial run, you may set load_pretrained_model to False to train the model yourself. This can take some time to finsih, depending the value you set for the epochs.

import pickle
load_pretrained_model = True
if load_pretrained_model:
    trained_model_file = open('./data/pretrained_model', 'rb')    
    db = pickle.load(trained_model_file)
    trained_model_file.close()
else:
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=16, warmup_steps=100)

samples = []
with open('sample.log', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            #emb = model.encode([line])
            samples.append(line)

# Create Embeddings and Upsert to Pinecone
emb = model.encode(samples)
prepped = []
for i in tqdm(range(len(samples))):
  v = {'id':f'{i}', 'values':emb[i].tolist(), 'metadata':{'log':samples[i]}}
  prepped.append(v)
index.upsert(prepped)

# Find the Anomaly
good_log_line = samples[0]
print(good_log_line)
results = []
while len(results)==0:  # After the upserts, it might take a few seconds for index to be ready for query.  
    time.sleep(2)       # If results is empty we try again two seconds later.
    queried = index.query(
        vector=emb[0].tolist(),
        include_metadata=True,
        top_k=100
    )
    results = queried['matches']
    print(".:. ",end="")
for i in range(0,10) :
  print(f"{round(results[i]['score'], 4)}\t{results[i]['metadata']['log']}")
last_element = len(results) -1 
print(f"{round(results[last_element]['score'], 4)}\t{results[last_element]['metadata']['log']}")

