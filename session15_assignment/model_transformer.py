import torch
import torch.nn as nn
import math
from pytorch_lightning import LightningModule, Trainer, Callback
import torchmetrics
from train import greedy_decode
from config import get_weights_file_path

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_sizes)
         # keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim=True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim=True) # (batch, seq_len, 1)
        #eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x -mean ) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_ff) ---> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_mode) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matric of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # create a vector of shape (d_model) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model)))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('PositionalEncoding X shape:', x.shape)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, 'embedding vector size d_model is not divisible by number of heads h'

        self.d_k = d_model // h # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # write a very low value (indicating -inf) to the position where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention score which can be used for visualization
        return (attention_scores @ value), attention_scores
    

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attentioin_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attentioin_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
                      d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blcoks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the decoder and encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

####*****------------------*************PYTORCH LIGHTNING CODE START

class transformerModel(LightningModule):
    def __init__(self, config, tokenizer_src, tokenizer_tgt, writer, num_examples):
        super().__init__()
        self.config = config
        # self.model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        self.model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                                       config['seq_len'], config['seq_len'], d_model=config['d_model'])
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.writer = writer
        self.num_examples = num_examples
        self.training_step_outputs = []

    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        # Run the tensors thru encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seqlen, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, )
        proj_output = self.model.project(decoder_output) # (B, seqlen, vocab_size)

        return proj_output
        
    def training_step(self, batch, batch_idx):
        
        encoder_input = batch['encoder_input'] # (b, seqlen)
        decoder_input = batch['decoder_input'] # (b, seqlen)
        encoder_mask = batch['encoder_mask'] # (B, 1, 1, seqlen)
        decoder_mask = batch['decoder_mask'] # (B, 1, seqlen, seqlen)

        proj_output = self(encoder_input, encoder_mask, decoder_input, decoder_mask)
        # Compare the output with the label
        label = batch['label'] # (B, seqlen)

        # Compute the loss using the simple cross entropy
        train_loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.training_step_outputs.append(train_loss)
        # train_loss = loss 
        values = {'train_loss': train_loss}
        self.log_dict(values, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return train_loss

    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_mean)
        # free up the memory
        self.training_step_outputs.clear()
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(self.config, f"{self.current_epoch:02d}")
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.trainer.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
            "global_step": self.global_step
        }, model_filename)
        print('Creating Checkpoint..')
        self.trainer.save_checkpoint(self.config["model_folder"]+"/s15Model.ckpt")


    def on_validation_epoch_start(self):
        # do something at the beginning of validation epoch...for example:
        self.model.eval()
        self.count = 0
    
        self.source_texts = []
        self.expected = []
        self.predicted = []
    
        try:
            # get the console width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                self.console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            self.console_width = 80

    def validation_step(self, batch, batch_idx):
        # # Run validation at the end of every epoch
        # run_validation(self.model, val_dataloader, self.tokenizer_src, self.tokenizer_tgt, self.config['seq_len'], 
        #                self.device, lambda msg: batch_iterator.write(msg), global_step, writer)

        self.count += 1
        encoder_input = batch['encoder_input'] # (b, seqlen)
        encoder_mask = batch['encoder_mask'] # (b, 1, 1, seqlen)

        # Check that the batch size is 1
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        
        model_output = greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, 
                                     self.tokenizer_tgt, encoder_input.size(1), self.device)

        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = self.tokenizer_tgt.decode(model_output.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

        # Print the source, target and model output
        print('-'*self.console_width)
        print(f"{f'SOURCE: ':>12}{source_text}")
        print(f"{f'TARGET: ':>12}{target_text}")
        print(f"{f'PREDICTED: ':>12}{model_out_text}")

        if self.count == self.num_examples:
            print('-'*self.console_width)
            # break
        # return val_loss
    
    # def test_step(self, batch, batch_idx):
    #     # Here we just reuse the validation_step for testing
    #     return self.validation_step(batch, batch_idx)
    
    # def predict_step(self, batch, batch_idx):
    #     X_batch, Y_batch = batch
    #     preds = self(X_batch)
    #     return preds    

    def on_validation_epoch_end(self):
        if self.writer:
            # Evaluate the character error rate
            # Compute the character error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(self.predicted, self.expected)
            self.writer.add_scalar('validation cer ', cer, self.global_step)
            self.writer.flush()
    
            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(self.predicted, self.expected)
            self.writer.add_scalar('validation wer ', wer, self.global_step)
            self.writer.flush()
    
            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(self.predicted, self.expected)
            self.writer.add_scalar('validation BLEU ', bleu, self.global_step)
            self. writer.flush()
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.config['lr'],
                                     eps=1e-9
                                     # weight_decay=config.WEIGHT_DECAY
                                    )
#         stepping_batches = self.trainer.estimated_stepping_batches
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
# #                                               max_lr=5.21E-04, 
#                                                         max_lr = self.config['lr'],
#                                                         pct_start=0.2,
#                                                         total_steps = stepping_batches,
# #                                               epochs=self.max_epochs, 
#                                                         div_factor=100.0, 
#                                                         final_div_factor=100.0, 
#                                                         anneal_strategy='linear',
# #                                               steps_per_epoch=self.steps_per_epoch
#                                                        )
        return ({'optimizer': optimizer, 
                 # 'lr_scheduler': {'scheduler': scheduler, 
                 #                  "monitor":"val_loss", 
                 #                  "interval":"step", 
                 #                  "frequency":1}
                })    
        
class saveCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        if pl_module.config['preload']:
            model_filename = get_weights_file_path(pl_module.config, f"{pl_module.config['preload']:02d}")
            print(f'Preloading Model {model_filename}')
            state = torch.load(model_filename)
            trainer.model.load_state_dict(state['model_state_dict'])
            pl_module.current_epoch = state['epoch'] + 1
            self.trainer.optimizers[0].load_state_dict(state['optimizer_state_dict'])
            self.pl_module.global_step = state['global_step']
            print("preloaded")
