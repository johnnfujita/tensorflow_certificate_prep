import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love my dog",
    "I love my cat",
    "You love my dog!",
    "Do you think my dog is amazing?"
]
## out of vocabulary###
###Uniformed sized sequences demands a paading

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
#### padding to the end, the maxlen=int() takes an int to define the maximum length of the sentence
### add truncating=str() post to loose information from the end

padded = pad_sequences(sequences, padding="post")
test_data = [
    "i really love my dog",
     "my dog loves my manatee"
]
test_seq = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_seq)
print(padded)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)