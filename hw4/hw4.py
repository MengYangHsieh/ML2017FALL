import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from util import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.3, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
parser.add_argument('--trainp', default = 'data/training_label.txt')
parser.add_argument('--trainup', default = 'data/training_nolabel.txt')
parser.add_argument('--testp', default = 'data/testing.txt')
args = parser.parse_args()

train_path = args.trainp
test_path = args.testp
semi_path = args.trainup
out_path = args.result_path

# build model
def simpleRNN(args):
    inputs = Input(shape=(args.max_length,))

    # Embedding layer
    embedding_inputs = Embedding(args.vocab_size, 
                                 args.embedding_dim, 
                                 trainable=True)(inputs)
    return_sequence = False
    dropout_rate = args.dropout_rate

    # RNN 
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size, 
                       return_sequences=return_sequence, 
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size, 
                        return_sequences=return_sequence, 
                        dropout=dropout_rate)
    RNN_output = RNN_cell(embedding_inputs)

    # BOW
    # bow_input = inputs
    # outputs = Dense(args.hidden_size,
    #                 activation="relu",
    #                 kernel_regularizer=regularizers.l2(0.1))(bow_input)
    # outputs = Dropout(dropout_rate)(outputs)

    # DNN layer
    outputs = Dense(args.hidden_size//2, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss=args.loss_function, optimizer=adam, metrics=[ 'accuracy',])
    
    return model

def main():

    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
    
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

    #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    else:
        # raise Exception ('Implement your testing parser')
        dm.read_data('test_data', test_path)
            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)
                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 

    # convert to sequences
    dm.to_sequence(args.max_length)

    # initial model
    print ('initial model...')
    model = simpleRNN(args)    
    print (model.summary())

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')

    # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )

    # testing
    elif args.action == 'test' :
        # raise Exception ('Implement your testing function')
        predicted_y = model.predict(dm.get_data('test_data'))
        print(predicted_y)
        with open(out_path, "w") as f:
            f.write("id,label\n")
            for i, y in enumerate(predicted_y):
                if y >= 0.5:
                    label = 1
                else:
                    label = 0
                f.write("{:d},{}\n".format(i, label))

    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)

        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        # repeat 10 times
        for i in range(10):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )

            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)

if __name__ == '__main__':
        main()
