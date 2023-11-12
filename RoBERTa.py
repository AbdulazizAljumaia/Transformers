#!/usr/bin/env python
# coding: utf-8

# In[1]:


# (0)
# Let's load the pre-trained model
from transformers import RobertaModel, RobertaTokenizer

# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)


# In[2]:


# (1)
# lets do the embedding
import torch

# Tokenize the words
input_ids_black = tokenizer("black", return_tensors="pt")
input_ids_brown = tokenizer("brown", return_tensors="pt")

# Get embeddings
with torch.no_grad():
    embedding_black = model(**input_ids_black).last_hidden_state
    embedding_brown = model(**input_ids_brown).last_hidden_state

# Since RoBERTa provides contextual embeddings, the output will be for each token in the sentence
# Usually, we take the embedding of the first token as the representation of the word
black_vec = embedding_black[0, 0, :]
brown_vec = embedding_brown[0, 0, :]


# In[3]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sklearn expects 2D array inputs
similarity = cosine_similarity([np.array(black_vec)], [np.array(brown_vec)])
print(f"Cosine Similarity: {similarity[0][0]}")


# In[4]:


len(brown_vec)


# In[19]:


# (2)
# The second step is to use the positional vector in attention model
# Define the dimension of the model
d_model = len(brown_vec)  # You might want to set this to the actual model dimension

# Initialize the positional encoding matrix
pe = torch.zeros(1, d_model)

import math
def positional_vector(pos):
    for i in range(0, len(brown_vec),2):
        pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe


# In[52]:


positional_vector(2)


# In[21]:


similarity = cosine_similarity([np.array(positional_encoding(2)[0])], [np.array(positional_encoding(10)[0])])
print(f"Cosine Similarity: {similarity[0][0]}")


# In[26]:


# (3)
# The second step is to use the positional [[[ encoding ]]] in attention model
y = torch.randn(1, d_model)  # This is an example initialization, adjust as needed

def positional_encoding(pos):
    pe = torch.zeros(1, d_model)  # Initialize the positional encoding tensor
    pc = torch.zeros(1, d_model)  # Initialize the pc tensor

    for i in range(0, len(brown_vec), 2):
        pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
        pc[0][i] = (y[0][i] * math.sqrt(d_model)) + pe[0][i]

        pe[0][i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
        pc[0][i + 1] = (y[0][i + 1] * math.sqrt(d_model)) + pe[0][i + 1]

    return pc


# In[27]:


similarity = cosine_similarity([np.array(positional_encoding(2)[0])], [np.array(positional_encoding(10)[0])])
print(f"Cosine Similarity: {similarity[0][0]}")


# In[28]:


# (4)
"""
Multi-head attention thing starts here. We need to get our hands dirty with these useless models that will not save humanities.
"""

import numpy as np
from scipy.special import softmax

# this time to reduce the computation process, we scale down the d_model to 4 instead of 1024
print("Step 1: Input : 3 inputs, d_model=4")
x =np.array([[1.0, 0.0, 1.0, 0.0], # Input 1
 [0.0, 2.0, 0.0, 2.0], # Input 2
 [1.0, 1.0, 1.0, 1.0]]) # Input 3
print(x)


# In[30]:


"""
Implementing QKV
Q to train queries
K to train keys
V to train values
"""

print("Step 2: weights 3 dimensions x d_model=4")
print("w_query")
w_query =np.array([[1, 0, 1],
 [1, 0, 0],
 [0, 0, 1],
 [0, 1, 1]])
print(w_query)

print("w_key")
w_key =np.array([[0, 0, 1],
 [1, 1, 0],
 [0, 1, 0],
 [1, 1, 0]])
print(w_key)


print("w_value")
w_value = np.array([[0, 2, 0],
 [0, 3, 0],
 [1, 0, 3],
 [1, 1, 0]])
print(w_value)


# In[32]:


print("Step 3: Matrix multiplication to obtain Q,K,V")
print("Query: x * w_query")
Q=np.matmul(x,w_query)
print(Q)

print("Key: x * w_key")
K=np.matmul(x,w_key)
print(K)

print("Value: x * w_value")
V=np.matmul(x,w_value)
print(V)


# In[41]:


print("Step 4: Scaled Attention Scores")
k_d = 1 #square root of k_d=3 rounded down to 1 for this example
attention_scores = (Q @ K.transpose())/k_d
print(attention_scores)


# In[42]:


print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0]=softmax(attention_scores[0])
attention_scores[1]=softmax(attention_scores[1])
attention_scores[2]=softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])


# In[43]:


print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
print("Attention 1")
attention1=attention_scores[0].reshape(-1,1)
attention1=attention_scores[0][0]*V[0]
print(attention1)
print("Attention 2")
attention2=attention_scores[0][1]*V[1]
print(attention2)
print("Attention 3")
attention3=attention_scores[0][2]*V[2]
print(attention3)


# In[44]:


print("Step7: summed the results to create the first line of the outputmatrix")
attention_input1=attention1+attention2+attention3
print(attention_input1)


# In[46]:


print("Step 8: Step 1 to 7 for inputs 1 to 3")
#We assume we have 3 results with learned weights (they were nottrained in this example)
#We assume we are implementing the original Transformer paper.We will have 3 results of 64 dimensions each
attention_head1=np.random.random((3, 64))
print(attention_head1)


# In[48]:


print("Step 9: We assume we have trained the 8 heads of the attentionsub-layer")
z0h1=np.random.random((3, 64))
z1h2=np.random.random((3, 64))
z2h3=np.random.random((3, 64))
z3h4=np.random.random((3, 64))
z4h5=np.random.random((3, 64))
z5h6=np.random.random((3, 64))
z6h7=np.random.random((3, 64))
z7h8=np.random.random((3, 64))
print("shape of one head",z0h1.shape,"dimension of 8 heads",64*8)


# In[49]:


print("Step 10: Concantenation of heads 1 to 8 to obtain the original 8x64=512 ouput dimension of the model")
output_attention=np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))
print(output_attention)


# In[53]:


# (5)
"""
Okay, the rest of the process is easy and it is repeated. Now, let's move to the actual work with hugging face. why is it called hugging face? I do not know.
"""

#@title Retrieve pipeline of modules and choose English to French translation
from transformers import pipeline


# In[81]:


translator = pipeline("translation_en_to_fr", model="t5-base", revision="686f1db")
#One line of code!
print(translator("It is easy to translate languages with transformers",
max_length=40))


# In[ ]:


# (6)
"""
OK, Aziz! Stop here and carry on with a new work :)
"""

