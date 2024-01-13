import LSTM
import torchtext
import torch


tokenizer = torchtext.data.get_tokenizer("basic_english")
vocab = torch.load('vocab.pth')


def predict_production(reviews, max_words=100):
    reviews_tokens = [vocab(tokenizer(text)) for text in reviews]
    reviews_tokens = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in reviews_tokens]
    X = torch.tensor(reviews_tokens, dtype=torch.int32)

    model = LSTM.LSTMRating(len(vocab), 50, 75, 2)
    model.load_state_dict(torch.load('LSTM.pth')['model'])

    with torch.no_grad():
        preds = model(X)
        preds_classes = preds.round().numpy()
        classes = ['Negative', 'Positive']
        for i in range(len(reviews)):
            print('{}- {} : {}'.format(i+1, reviews[i], classes[int(preds_classes[i][0])]))



# Testing model prediction in production
ask_user = True
reviews = []
print('Hello and welcome !!!')
print('-----------------------------------------------')
while ask_user:
    review = input('Enter the review to test the model : ')
    reviews.append(review)
    again = input('Would you like to enter another review (y = Yes, n= No) : ')
    if again.lower() != 'y':
        ask_user = False
    print('-----------------------------------------------')

print('\nPredicting:')
predict_production(reviews)
