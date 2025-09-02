import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BillingualDataset, casual_mask
from datasets import load_dataset
from config import get_weights_file_path, get_config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from transformer import build_transformer
from tqdm import tqdm
import warnings
import gc

from torch.utils.tensorboard import SummaryWriter

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Loading dataset
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, test_ds_size])

    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # OPTIMIZATION 1: Increase batch size and optimize DataLoader
    # Colab T4 can handle larger batch sizes for transformers
    optimized_batch_size = min(config.get('batch_size', 32) * 2, 64)  # Double batch size up to 64
    
    train_dataloader = DataLoader(
        train_ds,
        batch_size=optimized_batch_size,
        shuffle=True,
        num_workers=4,          # Increased from 2 to 4 for better parallelization
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4        # Prefetch more batches
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,           # No need to shuffle validation
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len,config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # OPTIMIZATION 2: Enable optimizations and set device properly
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    # Enable GPU optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # OPTIMIZATION 3: Use torch.compile for PyTorch 2.0+ (if available)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("Using torch.compile for optimization")
    except:
        print("torch.compile not available, continuing without it")

    # Tensorboard for loss visualization
    writer = SummaryWriter(config['experiment_name'])

    # OPTIMIZATION 4: Use AdamW with better hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(0.9, 0.98),  # Better betas for transformers
        eps=1e-9,
        weight_decay=0.01   # Small weight decay for regularization
    )
    
    # OPTIMIZATION 5: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.01
    )

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # OPTIMIZATION 6: Use gradient accumulation for effective larger batch size
    gradient_accumulation_steps = 2
    effective_batch_size = train_dataloader.batch_size * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")

    # OPTIMIZATION 7: Use automatic mixed precision (AMP)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    use_amp = device == "cuda"

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        # OPTIMIZATION 8: Reset gradients once per accumulation cycle
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            # OPTIMIZATION 9: Use automatic mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.proj(decoder_output)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                
                scaler.scale(loss).backward()
            else:
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.proj(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    # OPTIMIZATION 10: Gradient clipping with AMP
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # OPTIMIZATION 11: Less frequent logging to reduce overhead
            if batch_idx % 10 == 0:  # Log every 10 batches instead of every batch
                batch_iterator.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:6.3f}"})
                writer.add_scalar('train_loss', loss.item() * gradient_accumulation_steps, global_step)

        # Step scheduler
        scheduler.step()
        
        # OPTIMIZATION 12: Periodic memory cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # OPTIMIZATION 12: Much less frequent model saving to reduce I/O
        # Only save every few epochs, not every single epoch
        if epoch % 3 == 0 or epoch == config['num_epochs'] - 1:  # Save every 3 epochs or final epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step
            }, model_filename)
            print(f"Model saved at epoch {epoch}")
        
        # OPTIMIZATION 13: Aggressive memory cleanup every epoch
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete
        gc.collect()

        print(f"Epoch {epoch} completed. LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Flush tensorboard less frequently
        if epoch % 5 == 0:
            writer.flush()

    writer.close()

# ADDITIONAL SPEED TIPS FOR COLAB:
def optimize_colab_settings():
    """Run this function before training for maximum Colab performance"""
    import os
    import torch
    
    # Set environment variables for optimal performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Check if we have a good GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize GPU settings
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Enable optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return True
    return False

# Call this before training
print("=== COLAB OPTIMIZATION SETUP ===")
has_good_gpu = optimize_colab_settings()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)