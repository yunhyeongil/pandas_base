https://wikidocs.net/86900

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 02:08:01 2021

@author: yun
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import re


import shutil
import os
import unicodedata
import urllib3
import zipfile



import tensorflow as tf


file_path = './fra.txt'
lines = pd.read_csv(file_path, names=['eng', 'fra', 'cc'], sep='\t')
print(lines.sample(5))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sos_token = '\t'
eos_token = '\n'
lines.fra = lines.fra.apply(lambda x: '\t' + x + '\n')
print('전체샘플의 수:',len(lines))
print(lines.sample(5))

num_samples = 3300

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    # 위에서 구현한 함수를 내부적으로 호출
    sent = unicode_to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    sent = re.sub(r"\s+", " ", sent)
    return sent

en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"
print(preprocess_sentence(en_sent))
print(preprocess_sentence(fr_sent).encode('utf-8'))

lines = lines[['eng','fra']][:50000]
lines.sample(5)

def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []

    with open("fra.txt", "r",encoding='UTF-8') as lins:
        #print("lins->",lins,"end")
        for i, lin in enumerate(lins):
            #print(list(enumerate(lins))[0:2])
            #print("lin->",lin)
            # source 데이터와 target 데이터 분리
            src_line, tar_line, _ = lin.strip().split('\t')
            #print("src_line->",src_line)
            #print("tar_line->",tar_line)
            # source 데이터 전처리
            src_line_input = [w for w in preprocess_sentence(src_line).split()]

            # target 데이터 전처리
            tar_line = preprocess_sentence(tar_line)
            tar_line_input = [w for w in ("<sos> " + tar_line).split()]
            tar_line_target = [w for w in (tar_line + " <eos>").split()]

            # src_line -> Go. tar_line -> Va !
            encoder_input.append(src_line_input)
            decoder_input.append(tar_line_input)
            decoder_target.append(tar_line_target)

            if i == num_samples - 1:
                break

    return encoder_input, decoder_input, decoder_target

sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()
print(sents_en_in[:5])
print(sents_fra_in[:5])
print(sents_fra_out[:5])

tokenizer_en = Tokenizer(filters="", lower=False) # 소문자로 이미 바꿔서 False
tokenizer_en.fit_on_texts(sents_en_in)
#print(tokenizer_en.index_word)
#print(tokenizer_en.word_index)
#print("before encoder->", sents_en_in)
encoder_input = tokenizer_en.texts_to_sequences(sents_en_in)
#print("after encoder->",encoder_input)
tokenizer_fra = Tokenizer(filters="", lower=False)
tokenizer_fra.fit_on_texts(sents_fra_in)
tokenizer_fra.fit_on_texts(sents_fra_out)
decoder_input = tokenizer_fra.texts_to_sequences(sents_fra_in)
decoder_target = tokenizer_fra.texts_to_sequences(sents_fra_out)

eng_vocab_size = len(tokenizer_en.word_index) + 1
fra_vocab_size = len(tokenizer_fra.word_index) + 1
print('영어단어장의 크기:', eng_vocab_size)
print('프랑스어 단어장의 크기:', fra_vocab_size)

max_eng_seq_len = max([len(line) for line in encoder_input])
max_fra_seq_len = max([len(line) for line in decoder_input])

print('영어 시퀀스의 최대길이:', max_eng_seq_len)
print('프랑스 시퀀스의 초대길이:', max_fra_seq_len)

print('전체샘플:', len(lines))
print('영어단어장의크기:',eng_vocab_size)
print('프랑스 단어장의 크기:', fra_vocab_size)
print('영어 시퀀스의 최대 길이:', max_eng_seq_len)
print('프랑스 시퀀스 최대 길이:', max_fra_seq_len)

encoder_input = pad_sequences(encoder_input, maxlen=max_eng_seq_len, padding="post")
decoder_input = pad_sequences(decoder_input, maxlen=max_fra_seq_len, padding="post")
decoder_target = pad_sequences(decoder_target, maxlen=max_fra_seq_len, padding="post")

print('영어 데이터의 크기(shape) :', np.shape(encoder_input))
print('프랑스어 입력데이터의 크기 : ', np.shape(decoder_input))
print('프랑스어 출력데이터의 크기 : ', np.shape(decoder_target))

#src_vocab_size = len(tokenizer_en.word_index) + 1
#tar_vocab_size = len(tokenizer_fra.word_index) + 1
eng_vocab_size = len(tokenizer_en.word_index) + 1
fra_vocab_size = len(tokenizer_fra.word_index) + 1
print("영어 단어 집합의 크기 : {:d}, 프랑스어 단어 집합의 크기 : {:d}".format(eng_vocab_size, fra_vocab_size))

print(encoder_input[0])

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

print('영어 데이터의 크기:',np.shape(encoder_input))
print('프랑스어 입력데이턱의 크기:', np.shape(decoder_input))
print('프랑스어 출력데이터 크기:',np.shape(decoder_target))

print("encoder input->",len(encoder_input[0][0]))
print("decoder_input->",decoder_input[0])
print("decoder_target->",decoder_target[0])

n_of_val = 990

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

print('영어 학습데이터의 크기 :', np.shape(encoder_input_train))
print('프랑스어 학습 입력데이터의 크기 :', np.shape(decoder_input_train))
print('프랑스어 학습 출력데이터의 크기 :',np.shape(decoder_target_train))

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# LSTM셀의 마지막 time step의 hidden state와 cell state를 디코더 LSTM의 첫번째 hidden state와 cell state전달해주자

encoder_inputs = Input(shape=(None, eng_vocab_size))
# 입력 텐서를 생성
encoder_lstm = LSTM(units= 256, return_state=True)
# hidden state 256인 LSTM을 생성
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# 디코더로 전달할 hidden state, cell state를 리턴. encoder_output은 여기서는 불필요.
encoder_states = [state_h, state_c]
# hidden state와 cell state를 다음 time step으로 전달하기 위해서 별도로 저장


decoder_inputs = Input(shape=(None, fra_vocab_size))
# 입력 텐서 생성
decoder_lstm = LSTM(units=256, return_sequences= True, return_state=True)
# hidden state size 256 디코더 LSTM 생성
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
# decoder output는 모든 timestep의 hidden state

decoder_softmax_layer = Dense(fra_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
model.summary()




------------------------------ 결과 --------------------------------------------------------------------------------------------          



runfile('C:/Users/yun.DESKTOP-S5DSL8N.000/양진욱/word.py', wdir='C:/Users/yun.DESKTOP-S5DSL8N.000/양진욱')
                                                      eng  ...                                                 cc
73863                           Your shirt is inside out.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #5...
176566  Can't you keep your dog from coming into my ga...  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...
150633             You were absent from school yesterday.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #6...
18576                                   I'll forgive you.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...
72798                           What a pleasant surprise!  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...

[5 rows x 3 columns]
전체샘플의 수: 189114
                                    eng  ...                                                 cc
38080              We're in the forest.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #8...
114810  There is no smoke without fire.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...
102993    You can't carry on like this.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...
40634             I couldn't finish it.  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #2...
45787            Are you all right now?  ...  CC-BY 2.0 (France) Attribution: tatoeba.org #3...

[5 rows x 3 columns]
have you had dinner ?
b'avez vous deja dine ?'
[['go', '.'], ['go', '.'], ['go', '.'], ['hi', '.'], ['hi', '.']]
[['<sos>', 'va', '!'], ['<sos>', 'marche', '.'], ['<sos>', 'bouge', '!'], ['<sos>', 'salut', '!'], ['<sos>', 'salut', '.']]
[['va', '!', '<eos>'], ['marche', '.', '<eos>'], ['bouge', '!', '<eos>'], ['salut', '!', '<eos>'], ['salut', '.', '<eos>']]
영어단어장의 크기: 820
프랑스어 단어장의 크기: 1615
영어 시퀀스의 최대길이: 5
프랑스 시퀀스의 초대길이: 12
전체샘플: 50000
영어단어장의크기: 820
프랑스 단어장의 크기: 1615
영어 시퀀스의 최대 길이: 5
프랑스 시퀀스 최대 길이: 12
영어 데이터의 크기(shape) : (3300, 5)
프랑스어 입력데이터의 크기 :  (3300, 12)
프랑스어 출력데이터의 크기 :  (3300, 12)
영어 단어 집합의 크기 : 820, 프랑스어 단어 집합의 크기 : 1615
[9 1 0 0 0]
영어 데이터의 크기: (3300, 5, 820)
프랑스어 입력데이턱의 크기: (3300, 12, 1615)
프랑스어 출력데이터 크기: (3300, 12, 1615)
encoder input-> 820
decoder_input-> [[0. 0. 1. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]]
decoder_target-> [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]]
영어 학습데이터의 크기 : (2310, 5, 820)
프랑스어 학습 입력데이터의 크기 : (2310, 12, 1615)
프랑스어 학습 출력데이터의 크기 : (2310, 12, 1615)
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None, 820)]  0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None, 1615)] 0                                            
__________________________________________________________________________________________________
lstm (LSTM)                     [(None, 256), (None, 1102848     input_1[0][0]                    
__________________________________________________________________________________________________
lstm_1 (LSTM)                   [(None, None, 256),  1916928     input_2[0][0]                    
                                                                 lstm[0][1]                       
                                                                 lstm[0][2]                       
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 1615)   415055      lstm_1[0][0]                     
==================================================================================================
Total params: 3,434,831
Trainable params: 3,434,831
Non-trainable params: 0
__________________________________________________________________________________________________







