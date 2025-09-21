from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import Counter
from typing import List


class Vocabulary:
    def __init__(self, glosses: List[List[str]]):
        
        self.special_tokens = ["<pad>", "<blank>", "<unk>", "<sos>", "<eos>"]
        
        self.str2int = self.build_vocab(glosses)
        self.int2str = {v: k for k, v in self.str2int.items()}
        
        self.pad_token   = self.str2int.get('<pad>', 0)
        self.blank_token = self.str2int.get('<blank>', 1)
        self.unk_token   = self.str2int.get('<unk>', 2)
        self.start_token   = self.str2int.get('<sos>', 3)
        self.end_token   = self.str2int.get('<eos>', 4)
        
        self.size = len(self.str2int)
        
        self.glosses = glosses

    def build_vocab(self, gloss_sequences: List[List[str]]) -> Dict[str, int]:
        
        gloss_set = set()
        
        for seq in gloss_sequences:
            gloss_set.update(seq)
        
        gloss_list = sorted(list(gloss_set)) 
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        vocab.update({gloss: idx + len(self.special_tokens) for idx, gloss in enumerate(gloss_list)})
        
        return vocab

    def encode(self, char_list: List[str]) -> List[int]:
        return [self.str2int.get(c, self.str2int['<unk>']) for c in char_list]

    def decode(self, int_list: List[int]) -> List[str]:
        return [self.int2str.get(i, '<unk>') for i in int_list]


    def display_histogram(self, top_k: int = 20, tick=True):

        # Aplatir la liste de glosses
        all_glosses = [gloss for sequence in self.glosses for gloss in sequence]

        # Compter la fréquence de chaque gloss
        gloss_counts = Counter(all_glosses)

        # Sélectionner les top_k gloss les plus fréquents
        most_common = gloss_counts.most_common(top_k)

        glosses, counts = zip(*most_common)

        # Affichage de l'histogramme
        plt.figure(figsize=(12, 6))
        plt.bar(glosses, counts, color='dodgerblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Histogram")
        plt.xlabel("Gloss")
        plt.ylabel("Frequency")
        plt.tight_layout()
        if not tick:
            plt.xticks([], [])
        plt.show()



class DataProcessor:
    def __init__(self, vocab: Vocabulary, model_type: str):
        self.encode = vocab.encode
        self.model_type = model_type
        self.vocab = vocab
    
    def chunk_video(self, frames: torch.Tensor, glosses: Dict[str, List[Tuple[int, int]]], chunk_size=512, stride=None, alignment=False):
        stride = stride if stride is not None else chunk_size
        n_frames = frames.shape[0]
        chunks = []

        for start in range(0, n_frames - chunk_size + 1, stride):
            end = start + chunk_size
            chunk_feats = frames[start:end]

            gloss_intervals = []

            for gloss, intervals in glosses.items():
                for s, e in intervals:
                    if e > start and s < end:
                        interval_start = max(s, start) - start
                        interval_end = min(e, end) - start
                        gloss_intervals.append((gloss, interval_start, interval_end))

            # Sort by start time
            gloss_intervals = sorted(gloss_intervals, key=lambda x: x[1])

            # Get infos
            relevant_glosses = [g for (g, s, e) in gloss_intervals]
            if alignment:
                timings = [(s, e) for (g, s, e) in gloss_intervals]
            else:
                timings = None

            chunks.append({
                'features': chunk_feats,
                'targets': relevant_glosses,
                'timings': timings,
            })

        return chunks


    def collate_fn(self, batch):
        batch_features = []
        batch_targets = []
        input_lengths = []
        target_lengths = []
        batch_timings = []

        for sample in batch:
            feats = sample['features']
            batch_features.append(feats)
            input_lengths.append(feats.shape[0])

            gloss_ids = torch.tensor(self.encode(sample['targets']), dtype=torch.long)
            batch_targets.append(gloss_ids)
            target_lengths.append(len(gloss_ids))

            batch_timings.append(sample.get('timings', None))

        padded_features = pad_sequence(batch_features, batch_first=True, padding_value=0.0)
        batch_size, src_len, _ = padded_features.size()

        src_key_padding_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
        for i, length in enumerate(input_lengths):
            src_key_padding_mask[i, :length] = False

        if self.model_type == 'ctc-transformer':
            targets_concat = torch.cat(batch_targets)
            return {
                'features': padded_features,
                'targets': targets_concat,
                'input_lengths': torch.tensor(input_lengths, dtype=torch.long),
                'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
                'src_key_padding_mask': src_key_padding_mask,
            }

        elif self.model_type == 'base-transformer':
            decoder_inputs = []
            decoder_targets = []

            max_tgt_len = 0
            for gloss_ids in batch_targets:
                decoder_input = torch.cat([
                    torch.tensor([self.vocab.start_token], dtype=torch.long),
                    gloss_ids
                ])
                decoder_target = torch.cat([
                    gloss_ids,
                    torch.tensor([self.vocab.end_token], dtype=torch.long)
                ])
                decoder_inputs.append(decoder_input)
                decoder_targets.append(decoder_target)
                max_tgt_len = max(max_tgt_len, decoder_input.size(0))

            padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.vocab.pad_token)
            padded_decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=self.vocab.pad_token)
            tgt_key_padding_mask = (padded_decoder_inputs == self.vocab.pad_token)

            # Attention mask based on timings
            mask_attention = torch.full((batch_size, max_tgt_len, src_len), float('-inf'))

            for i, timings in enumerate(batch_timings):
                if timings is None:
                    mask_attention[i].fill_(0.0) 
                    continue

                mask_attention[i].fill_(float('-inf'))  
                mask_attention[i, 0, :].fill_(0.0)     

                for t_idx, (start_frame, end_frame) in enumerate(timings):
                    if t_idx + 1 >= max_tgt_len:
                        break
                    start_expanded = max(0, start_frame - self.mask_margin)
                    end_expanded = min(src_len, end_frame + self.mask_margin)
                    mask_attention[i, t_idx + 1, start_expanded:end_expanded] = 0.0
              

            return {
                'features': padded_features,
                'decoder_inputs': padded_decoder_inputs,
                'decoder_targets': padded_decoder_targets,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'attention_mask': mask_attention,  
            }


    def build_dataloader(self, dataset, batch_size=8, shuffle=True):
        def collate(batch):
            return self.collate_fn(batch)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    def create_loaders(self, data: Dict[str, List[List]], 
                       batch_size=8, chunk_size=512, stride=None, shuffle=True, alignment=False, mask_margin=0, train_ratio=0.8, val_ratio=0.1):
        
        self.mask_margin = mask_margin
        dataset = []
        for frames, glosses in zip(data['features'], data['gloss']):
            
            chunks = self.chunk_video(frames, glosses, chunk_size, stride, alignment)
                
            dataset.extend(chunks)

        total = len(dataset)
        train_size = int(train_ratio * total)
        val_size = int(val_ratio * total)
        test_size = total - train_size - val_size

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        train_loader = self.build_dataloader(train_set, batch_size=batch_size, shuffle=shuffle)
        val_loader = self.build_dataloader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = self.build_dataloader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
