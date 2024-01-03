import torch


class LSTMRating(torch.nn.Module):
    def __init__(self, vocab_len, embed_len, hidden_dim, n_layers):
        super(LSTMRating, self).__init__()
        self.vocab_len = vocab_len
        self.embed_len = embed_len
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_len)
        self.lstm = torch.nn.LSTM(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, dropout=0.1, batch_first=True)
        self.linear1 = torch.nn.Linear(hidden_dim, 24)
        self.linear2 = torch.nn.Linear(24, 1)


    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        hidden, carry = torch.randn(self.n_layers, len(X_batch), self.hidden_dim), torch.randn(self.n_layers, len(X_batch), self.hidden_dim)
        output, (hidden, carry) = self.lstm(embeddings, (hidden, carry))
        L_output1 = torch.relu_(self.linear1(output[:,-1,:]))
        return torch.sigmoid(self.linear2(L_output1))
