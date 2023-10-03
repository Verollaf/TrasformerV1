from TrasformerV1.model.attentione.scaled_dot_product_attention import ScaleDotProductAttention
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. Calcolo il prodotto scalare con le matrici dei pesi
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Divido il tensore in base al numero di testate (heads)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Eseguo il prodotto scalare scalato per calcolare la similarità
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. Concateno e passo al layer lineare
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. Visualizzo la mappa di attenzione
        # self.visualize_attention(out, q, k)

        return out

    def split(self, tensor):
        """
        Divido il tensore in base al numero di teste (heads).

        :param tensor: [batch_size, lunghezza, d_model]
        :return: [batch_size, head, lunghezza, d_tensor]
        """
        batch_size, lunghezza, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, lunghezza,
                             self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Funzione inversa di self.split(tensor : torch.Tensor).

        :param tensor: [batch_size, head, lunghezza, d_tensor]
        :return: [batch_size, lunghezza, d_model]

        """
        batch_size, head, lunghezza, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(
            batch_size, lunghezza, d_model)
        return tensor

    def visualize_attention(attention_map, input_tokens, output_tokens):
        """
        Visualizza una mappa di attenzione.

        :param attention_map: Un tensore PyTorch rappresentante la mappa di attenzione.
        :param input_tokens: Lista di token dell'input.
        :param output_tokens: Lista di token dell'output.

        """
        # Converti il tensore PyTorch in un array NumPy
        attention_map = attention_map.detach().cpu().numpy()
        input_tokens = input_tokens.detach().cpu().numpy()
        output_tokens = output_tokens.detach().cpu().numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(attention_map, cmap='viridis', aspect='auto')

        # Etichette degli assi
        plt.xticks(range(len(output_tokens)), output_tokens, rotation=45)
        plt.yticks(range(len(input_tokens)), input_tokens)

        # Etichette degli assi
        plt.xlabel("Output Tokens")
        plt.ylabel("Input Tokens")

        plt.title("Mappa di Attenzione")
        plt.colorbar()

        plt.tight_layout()
        plt.show()
