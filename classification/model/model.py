import torch.nn as nn
import torch


class IntentClassifierNLI(nn.Module):
    def __init__(self, base_model, hidden_size=768, dropout=0.5):
        super(IntentClassifierNLI, self).__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.similarity_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def encode(self, pairs, attention_masks):
        """
        :param pairs: intent x utterance pair ids (bs, n_pairs, max_len)
        :param attention_masks: intent x utterance pair attention masks (bs, n_pairs, max_len)
        :return: pair embeddings (bs, n_pairs, emb_size)
        """
        bs, n_pairs, seq_len = pairs.size()
        flat_pairs = pairs.view(-1, seq_len)  # (batch_size*n_pairs, max_len)
        attention_masks = attention_masks.view(-1, seq_len)

        output = self.base_model(flat_pairs.long(), attention_masks.long(),output_hidden_states=True).hidden_states[-1]  # (batch_size*n_pairs,max_len,emb_size)
        s_tokens = output[:,0,:]

        return s_tokens.reshape(bs, n_pairs, -1)  # (batch_size, n_pairs, emb_size)

    def mlmForward(self, mask_input_ids,right_attention_mask, Y):
        # BERT forward
        outputs = self.base_model(input_ids=mask_input_ids,attention_mask=right_attention_mask,labels=Y)
        return outputs.loss

    def embeddingContrastiveForword(self,uttr_only_ids,uttr_only_attention_mask,right_input,attention_mask):
        # (batch_size, emb_size)
        right_embedding = self.base_model(input_ids=right_input.long(), attention_mask=attention_mask.long(),
                                          output_hidden_states=True).hidden_states[-1][:, 0, :].unsqueeze(
            1)  # (batch_size,1,emb_size)
        cls_tokens_uttr_only = self.encode(uttr_only_ids, uttr_only_attention_mask)

        cos_sim=torch.cosine_similarity(cls_tokens_uttr_only,right_embedding,dim=-1)

        return cos_sim

    def forward(self, pairs, attention_masks):
        # (batch_size, n_pairs, emb_size)
        cls_tokens = self.encode(pairs, attention_masks)

        # batch_size, n_pairs
        similarity = self.similarity_layer(cls_tokens).squeeze(dim=2)  # (batch_size, n_pairs)

        return similarity


class PromptEncoder(torch.nn.Module):
    def __init__(self,hidden_size,softprompt_length,device):
        super().__init__()
        self.hidden_size = hidden_size
        self.device=device

        self.softprompt_length =softprompt_length
        self.seq_indices = torch.LongTensor(list(range(self.softprompt_length))).to(self.device)

        # softprompt的embedding
        self.embedding = torch.nn.Embedding(softprompt_length, self.hidden_size).to(self.device)

        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,

                                       dropout=0.5,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds

class SoftpromptIntentClassifierNLI(nn.Module):
    
    def __init__(self, IntentClassifier, hidden_size=768, dropout=0.5,softprompt_length=6,device=None):
        super(SoftpromptIntentClassifierNLI, self).__init__()
        self.hidden_size = hidden_size

        self.softprompt_length=softprompt_length
        self.device=device

        # load prompt encoder
        self.hidden_size = 768
        self.prompt_encoder = PromptEncoder(self.hidden_size,self.softprompt_length,self.device)
        self.IntentClassifier=IntentClassifier
        # frozen base model
        for param in self.IntentClassifier.base_model.parameters():
            param.requires_grad = False
        self.embeddings = self.IntentClassifier.base_model.get_input_embeddings()


        # for param in self.IntentClassifier.similarity_layer.parameters():
        #     param.requires_grad = False

    

    def embed_input(self,pairs,attention_masks):
        # pairs: (batch_size,n_pairs,max_len)
        bs, n_pairs, seq_len = pairs.size()

        flat_pairs = pairs.view(-1, seq_len)  # (batch_size*n_pairs, max_len)
        # (batch_size*n_pairs,max_len,hidden_size)=>
        # (batch_size,n_pairs,max_len,hidden_size)
        raw_embeds = self.embeddings(flat_pairs).view(bs, n_pairs, seq_len,-1)

        # softprompt(softprompt_length,hidden_size)
        replace_embeds = self.prompt_encoder()
        #(softprompt_length,hidden_size)=>(1,1,softprompt_length,hidden_size)=>
        # (batch_size,n_pairs,softprompt_length,hidden_size)
        raw_embeds_expanded=replace_embeds.unsqueeze(0).unsqueeze(0).expand(bs,n_pairs,self.softprompt_length,self.hidden_size)

        inputs_embeds = torch.cat((raw_embeds[:,:,0,:].unsqueeze(2), raw_embeds_expanded, raw_embeds[:,:,0:,:]),dim=2)


        # (batch_size,n_pairs,softprompt_length)
        prompt_mask=torch.ones(bs, n_pairs,self.softprompt_length).to(self.device)

        # attention_masks:(batch_size, n_pairs, max_len)=>(batch_size,n_pairs,softprompt_length+max_len)
        prompt_attention_masks=torch.cat((attention_masks[:,:,0].unsqueeze(2), prompt_mask, attention_masks[:,:,0:]),dim=2)


        # (batch_size,n_pairs,softprompt_length+max_len,hidden_size)=》(batch_size*n_pairs,softprompt_length+max_len,hidden_size)
        inputs_embeds_flatted=inputs_embeds.view(bs*n_pairs,-1,self.hidden_size)

        # (batch_size,n_pairs,softprompt_length+max_len)=>(batch_size*n_pairs,softprompt_length+max_len)
        prompt_attention_masks_flatted=prompt_attention_masks.view(bs*n_pairs,-1)

        return inputs_embeds_flatted,prompt_attention_masks_flatted


    def encode(self, pairs, attention_masks):
        bs, n_pairs, seq_len = pairs.size()

        inputs_embeds,prompt_attention_masks = self.embed_input(pairs,attention_masks)

        output = self.IntentClassifier.base_model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=prompt_attention_masks.to(self.device).bool(), output_hidden_states = True
                            ).hidden_states[-1]

        cls_tokens = output[:,0,:]
        return cls_tokens.reshape(bs, n_pairs, -1)  # (batch_size, n_pairs, emb_size)


    def forward(self, pairs, attention_masks):
        """
        :param pairs: intent x utterance pair ids (bs, n_pairs, max_len)
        :param attention_masks: intent x utterance pair attention mask (bs, n_pairs, max_len)
        :return: pair similarities (bs, n_pairs)
        """
        cls_tokens = self.encode(pairs, attention_masks)
        similarity = self.IntentClassifier.similarity_layer(cls_tokens).squeeze(dim=2)  # (batch_size, n_pairs)

        return similarity