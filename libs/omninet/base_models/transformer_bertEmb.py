import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None,k_gate=None):
        attn = torch.bmm(q, k.transpose(1, 2)) # [(n*b) x lq x dk] dot_product [(n*b) x dk x lk] = [(n*b) x lq * lk]
        attn = attn / self.temperature
        if k_gate is not None:
            attn=torch.mul(attn,k_gate)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn) # row-wise softmax
        attn = self.dropout(attn) 
        output = torch.bmm(attn, v) # [(n*b) x lq * lk] dot_product [(n*b) x lv x dv] (where lk == lv) = [(n*b) x lq x dv]

        return output, attn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_model//n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, k_gate=None):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        if k_gate is not None:
            k_gate = k_gate.transpose(0, 1)
            k_gate=k_gate.reshape(n_head*sz_b,len_q,len_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        #A Weighting score for the keys is provided
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask,k_gate=k_gate)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        attn=attn.view(n_head,sz_b,len_q,len_v).transpose(0,1)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        h, attn = self.attn(x,x,x,mask=attn_mask)
        h = self.feed_forward(h)
        return h, attn

class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.

    Args:
        vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len    (int)    : input sequence length
        d_model    (int)    : number of expected features in the input
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
        pad_id     (int)    : pad token id

    Examples:
    >>> encoder = TransformerEncoder(vocab_size=1000, seq_len=512)
    >>> inp = torch.arange(512).repeat(2, 1)
    >>> encoder(inp)
    """
    
    def __init__(self, seq_len, d_model=512, n_layers=6, n_heads=8, p_drop=0.1, d_ff=2048, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model) # (seq_len+1, d_model)

        # layers
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        # layers to classify
        # self.linear = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(p=p_drop)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, src_mask):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device).repeat(inputs.size(0), 1) + 1
        # position_pad_mask = inputs.eq(self.pad_id)
        # positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        # outputs = self.embedding(inputs) + self.pos_embedding(positions)
        outputs = inputs + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)
        outputs = self.dropout(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        # attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs, src_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)
        
        # outputs, _ = torch.max(outputs, dim=1)
        # # |outputs| : (batch_size, d_model)
        # outputs = self.linear(outputs)
        # # |outputs| : (batch_size, 2)

        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)

class TransformerBertEmb(nn.Module):
    def __init__(self, num_classes, bert, d_word_vec, n_layers, n_head, 
                 d_model, d_inner, dropout=0.1, n_position=200, pad_id=0):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(d_word_vec, d_model)
        self.encoder = TransformerEncoder(n_position, 
            d_model=d_model, n_layers=n_layers, n_heads=n_head, 
            p_drop=dropout, d_ff=d_inner, pad_id=pad_id)
        self.clf_head = nn.Linear(d_model, num_classes)
    
    def forward(self, x, padding_mask=None):
        with torch.no_grad():
            x = self.bert(x)[0]

        hidden_states, attention_weights = self.encoder(self.fc(x), padding_mask)
        return hidden_states, attention_weights

def TransformerBertEmb_IMDB(pretrained, bert, pad_id=None):
    d_word_vec = 768
    n_layers = 2
    n_head = 2
    d_model = 64
    d_inner = 128
    n_position = 512
    dropout = 0.1

    model = TransformerBertEmb(2, bert, d_word_vec, n_layers, n_head, d_model, d_inner, dropout=dropout, n_position=n_position, pad_id=pad_id)
        
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(pretrained))
    return model
