import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import json
import sys
import torchtext
import warnings as wrn
import LSTM
from sklearn.metrics import confusion_matrix, accuracy_score
wrn.filterwarnings('ignore')


'''
Model Hyper-Parameters :
1- Number of layers of LSTM
2- Number of hidden cells in LSTM
3- Number of LSTMs (stacked)
4- Bi-directional LSTM ?
5- Dropout
6- regularization
7- Number of linear layers
8- Activation functions
9- Loss function
10- Optimizer
11- Learning Rate
12- Epochs

We trained the model on 4.5 million reviews only as the computer could not handle the whole dataset
'''


tokenizer = torchtext.data.get_tokenizer("basic_english")
vocab = torch.load('vocab.pth')
mini_epochs_test = [0, 75000, 125000, 250000, 350000, 425000, 525000, 550000, 650000, 775000, 825000, 950000
, 1050000, 1125000, 1175000, 1250000, 1300000, 1375000, 1450000, 1550000, 1600000, 1750000, 1825000, 1975000, 2050000, 2175000, 2200000, 2350000
, 2450000, 2575000, 3000000, 3150000, 3425000, 3750000, 4125000, 4275000]


def handle_large_jsons(size, offset, filename="../yelp_dataset/yelp_academic_dataset_review.json", columns=['review_id', 'stars', 'text']):
    '''
    parameters :
    size : number of json objects/lines to process
    offset : line index to start from

    - open the large json file
    - go through it line by line and load the target data

    Return :
    dataframe with text, stars, review_id
    '''
    df_data = []
    with open(filename) as file:
        counter = -1
        for line in file:
            counter += 1
            if(counter < offset):
                continue
            data = json.loads(line.replace('\n', ''))
            row = []
            for col in columns:
                row.append(data[col])
            df_data.append(row)
            if counter == size+offset-1:
                break
    return pd.DataFrame(data=df_data, columns=columns)



def create_vocab(text_data):
    # creates vocabulary from text corpus and saves result in .pth file
    vocab = torchtext.vocab.build_vocab_from_iterator(generate_tokens(text_data), min_freq=1, specials=["<NA>"])
    vocab.set_default_index(vocab["<NA>"])
    torch.save(vocab, 'vocab.pth')
    print('Built and saved vocab ...')



def generate_tokens(text_data):
    # tokenize text data
    for text in text_data:
        yield tokenizer(text)



def vectorize_batch(batch, max_words=100):
    '''
    The zip() function takes n iterables, and returns y tuples, where y is the least of the length of all of the iterables provided.
    The yth tuple will contain the yth element of all of the iterables provided.
    For example:
    zip(['a', 'b', 'c'], [1, 2, 3])
    Will yield
    ('a', 1) ('b', 2) ('c', 3)

     n python, * is the 'splat' operator. It is used for unpacking a list into arguments.
     For example: foo(*[1, 2, 3]) is the same as foo(1, 2, 3).


     Parameters : a batch of X and Y

     - Tokeize every review in X
     - Padd X to be max_words

     Returns:
     Two tensors X and Y
    '''
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    # Padding inputs to be of size "max_words"
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    return torch.tensor(X, dtype=torch.int32), torch.tensor(Y) - 1



def create_dataloader(dataset, is_training):
    return torch.utils.data.DataLoader(dataset, batch_size=1024, collate_fn=vectorize_batch, shuffle=is_training)


def load_model(is_initial, model_name='LSTM.pth'):
    print('Loading model ...')
    model = LSTM.LSTMRating(len(vocab), 50, 75, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if is_initial == False:
        checkpoint = torch.load(model_name)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epochs'], checkpoint['training_losses'], model, optimizer

    else:
        return model, optimizer



def evaluate_model(model, loss_function, testing_data):
    '''
    Test the model by Passing X and Y
    and calculating the :
    1- Loss
    2- accuarcy score
    3- F1 score
    '''
    with torch.no_grad():
        print('Mini Epoch started : {}\n'.format(datetime.now()))
        Y_shuffled, Y_preds, losses = [], [], []
        batch_counter = 1
        for X, Y in testing_data:
            print('--- Batch {} ---'.format(batch_counter))
            preds = model(X)
            Y = Y.unsqueeze(1).float()
            loss = loss_function(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds)

            batch_counter+=1

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        # Saving evaluation metrics
        print("Validation Loss : {}".format(np.array(losses).mean()))
        # what is detach ????
        print("Accuracy score  : {}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.round().detach().numpy())))
        tn, fp, fn, tp = confusion_matrix(Y_shuffled.detach().numpy(), Y_preds.round().detach().numpy()).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        print("Precision : {}, Recall : {}, F1 : {}".format(precision, recall, f1))
        print('\nMini Epoch Endded : {}'.format(datetime.now()))

        return np.array(losses).mean(), accuracy_score(Y_shuffled.detach().numpy(), Y_preds.round().detach().numpy()), f1


def train_model(model, loss_function, optimizer, training_data):
    '''
    1- Make predictions
    2- calculate loss
    3- clear previously calculated gradients
    4- calculates Gradients
    5- Update model parameters weights
    '''
    print('Mini Epoch started : {}\n'.format(datetime.now()))
    losses = []
    batch_counter = 1
    for X, Y in training_data:
        print('--- Batch {} ---'.format(batch_counter))
        Y_preds = model(X)
        Y = Y.unsqueeze(1).float()
        loss = loss_function(Y_preds, Y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_counter+=1

    # Saving evaluation metrics
    print("Training Loss : {}".format(np.array(losses).mean()))
    print('Mini Epoch Endded : {}'.format(datetime.now()))
    return np.array(losses).mean()



def testing_phase(size):
    '''
    Perform the complete testing phase
    1- Load the model
    2- load a 25 thousand reviews
    3- prepare the data
    4- perform testing
    5- append the results to the list of results
    6- repeat steps 2 through 5 until we run out of reviews
    '''
    testing_losses, testing_accuracy, testing_f1 = [], [], []

    epochs, training_losses, model, optimizer = load_model(False)
    print('Epoch Number : ', epochs)

    batch_counter = 0
    for offset in mini_epochs_test:
        print('\n------------------------\nTesting : ', (batch_counter+1),'\n------------------------')
        test_df = handle_large_jsons(size, offset)
        test_df['stars'] = test_df['stars'].astype(int)
        test_df['sentiment'] = test_df['stars'].apply(lambda x : 1 if x <= 3 else 2)

        test_dataset = list(zip(test_df.iloc[:, 3], test_df.iloc[:, 2]))
        test_loader = create_dataloader(test_dataset, False)

        loss, accuarcy, f1 = evaluate_model(model, torch.nn.BCELoss(), test_loader)

        testing_losses.append(loss)
        testing_accuracy.append(accuarcy)
        testing_f1.append(f1)
        batch_counter +=1

    torch.save({'testing_losses': testing_losses, 'testing_accuracy': testing_accuracy, 'testing_f1': testing_f1
    , 'epochs': epochs}, 'LSTM_Evaluation.pth')

    print('Saved Evaluation Results ...')
    print('---- End ----')



def training_phase(size, is_initial):
    '''
    Perform the complete training phase
    1- Load the model
    2- load a 25 thousand reviews
    3- prepare the data
    4- perform training
    5- append the results to the list of results
    6- repeat steps 2 through 5 until we run out of reviews
    '''
    model, optimizer = None, None
    epochs = 0
    training_losses = []

    if is_initial:
        model, optimizer = load_model(is_initial)
    else:
        epochs, training_losses, model, optimizer = load_model(is_initial)

    print('Epoch Number : ', epochs)

    batch_counter = 0
    for offset in range(0, 4500000, size):
        if offset in mini_epochs_test :
            continue
        print('\n------------------------\nTraining : ',(batch_counter+1),'\n------------------------')
        train_df = handle_large_jsons(size, offset)
        train_df['stars'] = train_df['stars'].astype(int)
        train_df['sentiment'] = train_df['stars'].apply(lambda x : 1 if x <= 3 else 2)

        train_dataset = list(zip(train_df.iloc[:, 3], train_df.iloc[:, 2]))
        train_loader = create_dataloader(train_dataset, True)
        training_losses.append(train_model(model, torch.nn.BCELoss(), optimizer, train_loader))
        batch_counter+=1


    torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict(), 'training_losses' : training_losses, 'epochs': epochs},'LSTM.pth')
    print('\nSaved model parameters ...')
    print('---- End ----')



def end_epoch():
    '''
    Save mean of
    - training loss
    - testing loss
    - accuarcy
    - F1 score

    Save model and metrics in .pth files
    '''
    # restarting training file
    epochs, tr_losses, model, optimizer = load_model(False)
    torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict(), 'training_losses' : [], 'epochs': epochs+1},'LSTM.pth')

    # restarting testing file
    test_checkpoint = torch.load('LSTM_Evaluation.pth')
    t_testing_losses = test_checkpoint['testing_losses']
    t_testing_accuracy = test_checkpoint['testing_accuracy']
    t_testing_f1 = test_checkpoint['testing_f1']
    t_epochs = test_checkpoint['epochs']

    torch.save({'testing_losses': [], 'testing_accuracy': [], 'testing_f1': []
    , 'epochs': epochs+1}, 'LSTM_Evaluation.pth')


    # epoch evaluation file
    training_losses, testing_losses, testing_accuracy, testing_f1 = [], [], [], []
    try:
        checkpoint = torch.load('epoch_metrics.pth')
        training_losses = checkpoint['training_losses']
        testing_losses = checkpoint['testing_losses']
        testing_accuracy = checkpoint['testing_accuracy']
        testing_f1 = checkpoint['testing_f1']
    except:
        print('caught ...')

    training_losses.append(np.array(tr_losses).mean())
    testing_losses.append(np.array(t_testing_losses).mean())
    testing_accuracy.append(np.array(t_testing_accuracy).mean())
    testing_f1.append(np.array(t_testing_f1).mean())

    torch.save({'training_losses': training_losses, 'testing_losses': testing_losses, 'testing_accuracy': testing_accuracy, 'testing_f1': testing_f1}, 'epoch_metrics.pth')
    print('\n------------------------\nEpoch Completed\n------------------------')
    print(torch.load('epoch_metrics.pth'))



def predict_production(reviews, max_words=100):
    reviews_tokens = [vocab(tokenizer(text)) for text in reviews]
    reviews_tokens = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in reviews_tokens]
    X = torch.tensor(reviews_tokens, dtype=torch.int32)

    _, _, model, _ = load_model(False)

    with torch.no_grad():
        preds = model(X)
        preds_classes = preds.round().numpy()
        classes = ['Negative', 'Positive']
        for i in range(len(reviews)):
            print('{}- {} : {}'.format(i+1, reviews[i], classes[int(preds_classes[i][0])]))



##################################################
print('---------------------------------------------------\nStarted ...')

# creating vocab : Run once to create and save vocab
# run once with the whole text dataset
# df = handle_large_jsons(4500000, 0)
# create_vocab(df.iloc[:, 2])

## Training section ##
is_initial = None
try :
    torch.load('LSTM.pth')
    is_initial = False
except :
    is_initial = True

training_phase(25000, is_initial)

## Testing section ##
testing_phase(25000)

# End and save
end_epoch()

'''
# Run the following to create a plot of testing accuarcy and f1 scores
metrics = torch.load('epoch_metrics.pth')
plt.rc('figure', figsize=(10, 8))
plt.plot(metrics['testing_accuracy'], linestyle='dotted', label='accuarcy')
plt.plot(metrics['testing_f1'], label='f1')
plt.legend(loc="upper left")
plt.xlabel("epoch")
plt.ylabel("validation metrics")
plt.xticks([0, 1, 2, 3, 4])
plt.savefig('../graphs/training_epochs.png')
'''

# Testing model prediction in production
'''
reviews = ['I think the design is old, boring and bland. as for the products, they have changed immensely', 'the hotel was great and the beach was perfect. The location is close to all of the best five start restaurants. however, they served very little amounts of food for breakfast']
predict_production(reviews)
'''
