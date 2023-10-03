import torch
from torch import nn

'''a
Classe che specifica l'encoder posizionale del modello
    
'''


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        '''

        :param d_model: dimensione del modello
        :param max_seq_len: lunghezza massima della sequenza        

        '''

        super().__init__()
        self.d_model = d_model

        # Crea la matrice di posizionamento
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(
            1)  # Crea un tensore di posizioni da 0 a max_seq_len - 1
        '''
        Calcola un termine divisorio div_term utilizzando la formula: div_term = exp(2 * i / d_model * -log(10000.0)), dove i varia da 0 a d_model con incrementi di 2. 
        Questo termine divisorio è utilizzato per generare le componenti sinusoidali e cosinusoidali delle informazioni posizionali.
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Assegna i valori pari
        # Assegna i valori dispari
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Aggiunge la matrice di posizionamento al batch
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
