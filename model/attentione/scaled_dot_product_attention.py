import math
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    """
    Calcola l'attenzione Scale Dot Product.

    Query : la frase su cui ci concentriamo (decoder).
    Key : ogni frase per verificare la relazione con Query (encoder).
    Value : ogni frase corrispondente a Key (encoder).

    N.B Ricorda esempio di Youtube:
    Query : cosa sto cercando
    Key : video nel database di Youtube
    Value : punteggio di ogni video nel database di Youtube (corrispondente a Key)


    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        # L'input è un tensore a 4 dimensioni
        # [batch_size, head, lunghezza, d_tensor]
        _, _, _, d_tensor = k.size()

        # 1. Calcola il prodotto scalare tra Query e Key^T per calcolare la similarità
        k_t = k.transpose(2, 3)  # Trasposizione
        score = (q @ k_t) / math.sqrt(d_tensor)  # Prodotto scalare scalato

        # 2. Applica la maschera (opzionale)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. Applica la funzione softmax per ottenere valori nell'intervallo [0, 1]
        score = nn.Softmax(score, dim=-1)

        # 4. Moltiplica per il Value
        v = score @ v

        return v, score
