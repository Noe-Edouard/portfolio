from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Dict, Tuple


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


class DataProcessor:
    def __init__(self, vocab: Vocabulary, model_type: str):
        self.encode = vocab.encode
        self.model_type = model_type
        self.vocab = vocab
    
    def chunk_video_no_alignment(self, frames: torch.Tensor, glosses: Dict[str, List[Tuple[int, int]]], chunk_size=512, stride=None):
        
        stride = stride if stride is not None else chunk_size
        
        n_frames = frames.shape[0]
        chunks = []

        for start in range(0, n_frames - chunk_size + 1, stride):
            end = start + chunk_size
            chunk_feats = frames[start:end]

            # Overlap timmings
            relevant_glosses = []
            for gloss, intervals in glosses.items():
                for s, e in intervals:
                    if e > start and s < end:
                        relevant_glosses.append(gloss)

            chunks.append({
                'features': chunk_feats,
                'targets': relevant_glosses
            })

        return chunks

    def chunk_video_alignment(self, frames_list: List[torch.Tensor], glosses: List[str], chunk_size=512) -> List[Dict]:
        chunks = []
        glosses_chunk = []
        features_chunk = []
        timings_chunk = []
        n_frames = 0
        position = 0  # Current frame in the all video

        for frames, gloss in zip(frames_list, glosses):
            frame_count = frames.shape[0]

            # Si on peut encore ajouter dans le chunk actuel
            if n_frames + frame_count <= chunk_size:
                features_chunk.append(frames)
                glosses_chunk.append(gloss)
                timings_chunk.append((position, position + frame_count))
                n_frames += frame_count
            else:
                # Fin du chunk actuel
                if n_frames > 0:
                    chunks.append({
                        'features': torch.cat(features_chunk, dim=0),
                        'targets': glosses_chunk,
                        'timings': timings_chunk,
                    })

                # On commence un nouveau chunk avec la séquence courante
                features_chunk = []
                glosses_chunk = []
                timings_chunk = []
                n_frames = frame_count

            position += frame_count

        # Dernier chunk s'il reste
        if n_frames > 0 and n_frames == chunk_size:
            chunks.append({
                'features': torch.cat(features_chunk, dim=0),
                'targets': glosses_chunk,
                'timings': timings_chunk,
            })

        return chunks


    def collate_fn(self, batch):
        batch_features = []
        batch_targets = []
        input_lengths = []
        target_lengths = []

        for sample in batch:
            feats = sample['features']
            batch_features.append(feats)
            input_lengths.append(feats.shape[0])
            
            gloss_ids = torch.tensor(self.encode(sample['targets']), dtype=torch.long)
            batch_targets.append(gloss_ids)
            target_lengths.append(len(gloss_ids))

        padded_features = pad_sequence(batch_features, batch_first=True)
        src_key_padding_mask = (padded_features.abs().sum(dim=-1) == 0)

        if self.model_type == 'ctc-transformer':
            # Targets are concatenated; lengths needed for CTC
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
            all_timings = []  # Liste des timings batchés
            
            for sample in batch:
                gloss_ids = torch.tensor(self.encode(sample['targets']), dtype=torch.long)
                # Ajouter sos/eos
                decoder_input = torch.cat([
                    torch.tensor([self.vocab.start_token]),
                    gloss_ids
                ])
                decoder_target = torch.cat([
                    gloss_ids,
                    torch.tensor([self.vocab.end_token])
                ])
                decoder_inputs.append(decoder_input)
                decoder_targets.append(decoder_target)
                all_timings.append(sample['timings'])  # timings est une liste de (start, end) par gloss

            padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.vocab.pad_token)
            padded_decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=self.vocab.pad_token)
            
            tgt_key_padding_mask = (padded_decoder_inputs == self.vocab.pad_token)

            # Construire le mask attention en fonction des timings
            batch_size, tgt_len = padded_decoder_inputs.shape
            src_len = padded_features.shape[1]

            attention_mask = torch.ones(batch_size, tgt_len, src_len, dtype=torch.bool)  # True = masked par défaut

            for b in range(batch_size):
                timings = all_timings[b]
                # décaler d'1 à cause du <sos> au début du decoder_input
                for t_idx, (start, end) in enumerate(timings, start=1):
                    # On démasque les frames correspondant aux timings du gloss t_idx-1
                    # Si t_idx dépasse la longueur cible on ignore
                    if t_idx >= tgt_len:
                        break
                    # Clamp pour rester dans les bornes src_len
                    s = max(0, start)
                    e = min(src_len, end)
                    attention_mask[b, t_idx, s:e] = False

                # <sos> (t_idx=0) peut rester masqué pour tout src (ou démasqué si tu préfères)
                attention_mask[b, 0, :] = False  # souvent on démasque <sos> partout

            return {
                'features': padded_features,
                'decoder_inputs': padded_decoder_inputs,
                'decoder_targets': padded_decoder_targets,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'attention_mask': attention_mask,  # masque attention ciblé à passer au modèle
            }


                
            
    

    def build_dataloader(self, dataset, batch_size=8, shuffle=True):
        def collate(batch):
            return self.collate_fn(batch)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    def create_loaders(self, data: Dict[str, List[List]], 
                       batch_size=8, chunk_size=512, stride=None, alignemnt=True, shuffle=True, train_ratio=0.8, val_ratio=0.1):
        dataset = []
        for frames, glosses in zip(data['features'], data['gloss']):
            
            if alignemnt:
                chunks = self.chunk_video_alignment(frames, glosses, chunk_size)
            else:
                chunks = self.chunk_video_no_alignment(frames, glosses, chunk_size, stride)
                
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
