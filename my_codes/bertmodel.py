import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import *


def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0)).to(A.device)
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)

class GCN(nn.Module):
    '''
    Z = AXW
    '''
    def __init__(self, dim_in=100, dim_hidden=100, dim_out=100):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)
        # self.fc1 = nn.Linear(dim_in ,dim_out,bias=False)

    def forward(self, A, X):
        '''
        计算两层gcn
        '''
        self.A = A
        X = F.relu(self.fc1(self.A.mm(X)))
        d = self.fc2(self.A.mm(X))
        return self.fc2(self.A.mm(X))
        # return X

class concept_gcn_lstm(nn.Module):
    def __init__(self, word_embed_dim, hidden_dim, lstm_out_dim, dropout_prob): # 注：句子的embedding和词语的embedding的长度可能不一样，注意这个word_embedding的转化
        super(concept_gcn_lstm, self).__init__()
        self.lstm_out_dim = lstm_out_dim
        self.hidden_dim = hidden_dim
        self.word_embed_dim = word_embed_dim
        self.lstm = nn.LSTM(word_embed_dim, lstm_out_dim)
        self.gcn = GCN(word_embed_dim, word_embed_dim, word_embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.l1 = nn.Linear(word_embed_dim * 2, word_embed_dim)

    def forward(self, question_embedding, concept_embedding, mask_size, adjacent_matrix_hat, qConcept_index, aConcept_index):
        # padding
        adjacent_matrix_hat = normalize(adjacent_matrix_hat, symmetric=True)
        concept_embedding = self.gcn(adjacent_matrix_hat, concept_embedding)
        out_q, _q = self.lstm(concept_embedding[qConcept_index])
        out_a, _a = self.lstm(concept_embedding[aConcept_index])
        lstm_out_q = _q[0][0]
        lstm_out_a = _a[0][0]
        QA_embed = torch.cat([question_embedding, lstm_out_a, lstm_out_q], dim=1)
        logits = self.l1(self.dropout(QA_embed))
        return logits


class BertForTask1(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTask1.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTask1, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size+100, config.num_labels)
        self.gcn = GCN(100, 100, 100)
        self.lstm = nn.LSTM(100, 100, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, guids=None, concept_embedding=None,
                adjacent_matrix=None, entity_index=None):

        # labels [batch]只有0/1   input_ids/attention_mask/token_type_ids [batch, choice, max_length]
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        token_type_ids = torch.stack(token_type_ids)
        num_sents = input_ids.size(1)
        assert num_sents == 2
        results = {}

        device = input_ids.device

        batch_size = input_ids.size(0)
        # Calculate each sent separately, and return loss * logits tuple containing 2 sent result
        logits_list = []
        loss_list = []
        for i in range(num_sents):
            input_ids_ = input_ids[:, i, :].squeeze(dim=1)
            attention_mask_ = attention_mask[:, i, :].squeeze(dim=1) if attention_mask is not None else None  # [batch, 1, max_length] --> [batch, max_length]
            token_type_ids_ = token_type_ids[:, i, :].squeeze(dim=1) if token_type_ids is not None else None

            # bert module
            outputs = self.bert(input_ids_,
                                attention_mask=attention_mask_,
                                token_type_ids=token_type_ids_)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

            # GCN module
            gcn_out = []
            for batch_id in range(batch_size):
                if len(adjacent_matrix[batch_id]) == 0:
                    gcn_out_single = torch.zeros(100).to(device)
                    gcn_out.append(gcn_out_single)
                else:
                    if len(entity_index[batch_id][i]) == 0:
                        gcn_out_single = torch.zeros(100).to(device)
                        gcn_out.append(gcn_out_single)
                    else:
                        adjacent_matrix_hat = normalize(adjacent_matrix[batch_id], symmetric=True)
                        X = concept_embedding[batch_id].to(torch.float32)
                        X = self.gcn(adjacent_matrix_hat, X)
                        out_, _ = self.lstm(X[entity_index[batch_id][i].cpu().numpy().tolist()].unsqueeze(0))
                        lstm_out = _[0][0][0]
                        gcn_out.append(lstm_out)
            gcn_out = torch.stack(gcn_out)

            # concatenate
            cat_embed = torch.cat([pooled_output, gcn_out], dim=1)

            logits = self.classifier(cat_embed)  # [batch,2]
            # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            logits_list.append(logits)

            if labels is not None: # Training的时候执行，计算loss
                loss_fct = CrossEntropyLoss()
                label_ = 1 - labels if i == 0 else labels # 原始label=0，则sent0错了，sent0_label=1，sent1_label=0.   原始label=1，则sent1错了，sent1_label=1, sent0_label=0
                loss = loss_fct(logits.view(-1, self.num_labels), label_.view(-1))  # 计算loss value, num_labels=22
                loss_list.append(loss)

        loss_sum = (loss_list[0] + loss_list[1]) / 2
        results.update({"logits": logits_list, "loss": loss_sum})

        return results




class BertForTask2(BertPreTrainedModel):
    """
    Labels: torch.LongTensor of shape (batch_size,)
        Labels for computing the multiple choice classification loss.
        Indices should be in [0, ..., num_choices] where num_choices is the size of the second dimension of input tensors.

    Outputs:
        loss: torch.FloatTensor of shape (1,), classification loss.
        classification_scores: torch.FloatTensor of shape (batch_size, num_choices), classification scores before Softmax.
        hidden_states: only returned when config.output_hidden_states=True. torch.FloatTensor.
            (one for the output of each layer + output of the embedding) of shape (batch_size, sequence_length, hidden_size).
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions: only returned when config.output_attention=True.
            (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            Attention weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForTask2.from_pretrained("bert-base-uncased")
        choices = ["I eat apples.", "I eat stones."]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # (Batch_size 1, 2 choices)
        labels = torch.tensor(1).unsqueeze(0)  # (Batch_size 1,)
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]
    """

    def __init__(self, config):
        super(BertForTask2, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, guids=None):
        # attention_mask [batch_size, num_choices, max_length]  input_ids [batch_size, num_choices, max_length]
        # token_type_ids [batch_size, num_choices, max_length]  labels [batch_size,]
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size, num_choices, max_length] --> [batch_size*num_choices, max_length]
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs[0] [batch_size*num_choices, max_length, hidden_size] ;   outputs[1] [batch_size*num_choices, hidden_size]

        pooled_output = outputs[1]  # [batch_size*num_choices, hidden_size]
        #拼向量
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size*num_choices, 1]
        reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here  # [batch_size, num_choices]

        if labels is not None: # 无label的test时，preds calculation要改
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # loss, reshaped_logits, (hidden_states), (attentions)


















