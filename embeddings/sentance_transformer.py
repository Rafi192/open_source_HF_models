from transformers.utils import logging
# code that suppresses warning messages.
logging.set_verbosity_error()

#-------------------------------------------------

#building the sentance embedding transformer

#--------------------------------------------------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLm-L6-v2")
# print(model)


sentances1 = [
    'The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome'
]

embeddings1 = model.encode(sentances1, convert_to_tensor=True)

# print(embeddings1)


sentances2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

embeddings2 = model.encode(sentances2, convert_to_tensor=True)


# now calculate cosine similarity between two sentancese as a measure of how similar they are to each other

from sentence_transformers import util

cosine_scores = util.cos_sim(embeddings1, embeddings2)

print("----------   cosine score -------",cosine_scores)