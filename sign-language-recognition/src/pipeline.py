import torch
import math
from time import time
from jiwer import wer
from tqdm import tqdm
import torch.nn.functional as F
from src.dataloader import DataProcessor, Vocabulary
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt 



class TransformerPipeline:
    def __init__(self, device, model_type='ctc-transformer'):
        self.device = device
        self.model_type = model_type
    
    
    # Create loaders and vocabulary
    def process_data(self, data, batch_size=16, chunk_size=512, stride=None, alignment=False, mask_margin=0, shuffle=True, train_ratio=0.8, val_ratio=0.1):
        
        self.vocabulary = Vocabulary(data['gloss'])
        self.dataloader = DataProcessor(self.vocabulary, self.model_type)
        
        train_loader, val_loader, test_loader = self.dataloader.create_loaders(
            data=data, 
            batch_size=batch_size, 
            chunk_size=chunk_size, 
            stride=stride,
            alignment=alignment,
            shuffle = shuffle,
            mask_margin=mask_margin,
            train_ratio=train_ratio, 
            val_ratio=val_ratio
        )
        
        return train_loader, val_loader, test_loader


    # Training model
    def train_model(self, model, train_loader, val_loader, epochs=10, early_stop=None, lr=1e-4):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if self.model_type == 'ctc-transformer':
            criterion = torch.nn.CTCLoss(blank=self.vocabulary.blank_token, zero_infinity=True)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary.pad_token)

        self.metrics = {
            'train_loss': [],
            'val_loss': [],
        }

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in progress_bar:
                features = batch['features'].to(self.device)
                src_mask = batch['src_key_padding_mask'].to(self.device)
                
                optimizer.zero_grad()

                if self.model_type == 'ctc-transformer':
                    targets = batch['targets'].to(self.device)
                    input_lengths = batch['input_lengths'].to('cpu')
                    target_lengths = batch['target_lengths'].to('cpu')

                    logits = model(features, src_key_padding_mask=src_mask)
                    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                else:  # base-transformer
                    tgt_input = batch['decoder_inputs'].to(self.device)
                    tgt_output = batch['decoder_targets'].to(self.device)
                    tgt_mask = batch['tgt_key_padding_mask'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    logits = model(
                        features=features,
                        targets=tgt_input,
                        src_key_padding_mask=src_mask,
                        tgt_key_padding_mask=tgt_mask,
                        memory_mask=attention_mask 
                    )
                    logits = logits.reshape(-1, logits.size(-1))
                    tgt_output = tgt_output.reshape(-1)

                    loss = criterion(logits, tgt_output)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

            train_loss = self.evaluate_model(model, train_loader)
            val_loss = self.evaluate_model(model, val_loader)

            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)

            if early_stop is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # torch.save(model.state_dict(), 'best_model.pt')
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stop:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            # else:
                # torch.save(model.state_dict(), 'best_model.pt')

        # model.load_state_dict(torch.load('best_model.pt'))
        return self.metrics




    # Evaluating model
    def evaluate_model(self, model, val_loader):
        model.eval()
        total_loss = 0.0
        count = 0

        if self.model_type == 'ctc-transformer':
            criterion = torch.nn.CTCLoss(blank=self.vocabulary.blank_token, zero_infinity=True)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary.pad_token)

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                src_mask = batch['src_key_padding_mask'].to(self.device)

                if self.model_type == 'ctc-transformer':
                    targets = batch['targets'].to(self.device)
                    input_lengths = batch['input_lengths'].to('cpu')
                    target_lengths = batch['target_lengths'].to('cpu')

                    logits = model(features, src_key_padding_mask=src_mask)
                    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                else:
                    tgt_input = batch['decoder_inputs'].to(self.device)
                    tgt_output = batch['decoder_targets'].to(self.device)
                    tgt_mask = batch['tgt_key_padding_mask'].to(self.device)

                    logits = model(features, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
                    logits = logits.reshape(-1, logits.size(-1))
                    tgt_output = tgt_output.reshape(-1)

                    loss = criterion(logits, tgt_output)

                total_loss += loss.item()
                count += 1

        return total_loss / count

    

    # Testing model
    def test_model(self, model, test_loader):
        model.eval()
        pred_sequences = []
        ref_sequences = []
        wer_sequence = []

        blank = self.vocabulary.blank_token
        start_token = self.vocabulary.start_token
        end_token = self.vocabulary.end_token

        def remove_consecutive_duplicates(seq):
            if not seq:
                return []
            cleaned = [seq[0]]
            for token in seq[1:]:
                if token != cleaned[-1]:
                    cleaned.append(token)
            return cleaned


        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                src_key_padding_mask = batch['src_key_padding_mask'].to(self.device)

                if self.model_type == 'ctc-transformer':
                    targets = batch['targets']
                    target_lengths = batch['target_lengths']

                    logits = model(features, src_key_padding_mask=src_key_padding_mask)
                    log_probs = F.log_softmax(logits, dim=-1)
                    preds = log_probs.argmax(dim=-1).cpu()

                    # slice and decode
                    start_idx = 0
                    for i, length in enumerate(target_lengths):
                        target_seq = targets[start_idx : start_idx + length].tolist()
                        start_idx += length

                        pred_seq = preds[i].tolist()
                        decoded = []
                        prev_token = None
                        for token in pred_seq:
                            if token != blank and token != prev_token:
                                decoded.append(token)
                            prev_token = token

                        pred_str = [self.vocabulary.int2str.get(idx, "<unk>") for idx in decoded]
                        ref_str = [self.vocabulary.int2str.get(idx, "<unk>") for idx in target_seq if idx != blank]

                        pred_sequences.append(pred_str)
                        ref_sequences.append(ref_str)

                        if not pred_str and not ref_str:
                            wer_value = 0.0
                        else: 
                            wer_value = wer(" ".join(ref_str), " ".join(pred_str)) 
                            
                        wer_sequence.append(wer_value)

                else:  # base-transformer
                    targets = batch['decoder_targets']
                    pred_ids = model.predict(features, start_token=start_token, end_token=end_token,
                                            src_key_padding_mask=src_key_padding_mask)

                    for i, pred_seq in enumerate(pred_ids):
                        target_seq = targets[i, :].tolist()
                        if end_token in target_seq:
                            target_seq = target_seq[:target_seq.index(end_token)]

                        pred_str = [self.vocabulary.int2str.get(idx, "<unk>") for idx in pred_seq if idx not in (start_token, end_token, self.vocabulary.pad_token)]
                        ref_str = [self.vocabulary.int2str.get(idx, "<unk>") for idx in target_seq if idx not in (start_token, end_token, self.vocabulary.pad_token)]

                        pred_str = remove_consecutive_duplicates(pred_str)
                        ref_str = remove_consecutive_duplicates(ref_str)

                        pred_sequences.append(pred_str)
                        ref_sequences.append(ref_str)
                        
                        if not pred_str and not ref_str:
                            wer_value = 0.0
                        else: 
                            wer_value = wer(" ".join(ref_str), " ".join(pred_str)) 
                        print('Wer : ', wer_value)
                        wer_sequence.append(wer_value)

        mean_wer = sum(wer_sequence) / len(wer_sequence) if wer_sequence else float('inf')
        print(f"Test WER: {mean_wer:.2f}")

        return mean_wer, pred_sequences, ref_sequences


