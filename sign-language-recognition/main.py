import torch
from src.models import BaseTransformer, CTCTransformer
from src.pipeline import TransformerPipeline
from src.analytics import display_confusion_matrix, display_metrics


# Load data
data = torch.load("../Test_dict.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize pipeline
pipeline = TransformerPipeline(device, 'base-transformer')
# pipeline = TransformerPipeline(device, 'ctc-transformer')

# Create loaders
train_loader, val_loader, test_loader = pipeline.process_data(
    data = data, 
    batch_size = 8,
    chunk_size = 256,
    stride = None,
    alignment = False,
    mask_margin=0,
    shuffle = True,
    train_ratio = 0.8,
    val_ratio = 0.1,
)

# Encoder-Decoder Model
model = BaseTransformer(
    input_dim = data["features"][0].shape[1],
    vocab_size = pipeline.vocabulary.size,
    d_model = 256,
    nhead = 4,
    num_layers = 2,
    dim_feedforward = 512,
    dropout = 0,
    max_len = 5000,
    use_cnn = True,
).to(device)

# CTC Model
# model = CTCTransformer(
#     input_dim = data["features"][0].shape[1],
#     vocab_size = pipeline.vocabulary.size,
#     d_model = 256,
#     nhead = 4,
#     num_layers = 2,
#     dim_feedforward = 512,
#     dropout = 0,
#     max_len = 5000,
#     use_cnn = True,
# ).to(device)


# Train model
metrics = pipeline.train_model(
    model = model, 
    train_loader = train_loader, 
    val_loader = val_loader, 
    epochs = 50, 
    early_stop = None,
    lr = 1e-4,
)



# Test model
wer_score, pred_sequences, ref_sequences = pipeline.test_model(model, test_loader)

display_metrics(metrics)
display_confusion_matrix(pred_sequences, ref_sequences)