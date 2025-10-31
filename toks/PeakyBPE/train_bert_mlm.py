import json
import torch
from pathlib import Path
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PickyBPETokenizerWrapper:
    """Wrapper to make PickyBPE compatible with Hugging Face"""
    
    def __init__(self, model_path: str):
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Extract vocabulary from the model
        self.vocab = {}
        self.ids_to_tokens = {}
        
        for token_data in model_data['tokens']:
            if token_data['present']:  # Only use present tokens
                token_id = model_data['id2int'][str(token_data['id'])]
                token_str = token_data['str']
                self.vocab[token_str] = token_id
                self.ids_to_tokens[token_id] = token_str
        
        self.vocab_size = len(self.vocab) + 4787
        
        # Get special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.mask_token = '<MASK>'  # You may need to add this
        self.cls_token = '<BOS>'  # Use BOS as CLS
        self.sep_token = '<EOS>'  # Use EOS as SEP
        
        logger.info(f"Loaded vocabulary with {self.vocab_size} tokens")


def create_tokenizer_from_pickybpe(pickybpe_model_path: str) -> PreTrainedTokenizerFast:
    """
    Create a Hugging Face tokenizer from PickyBPE model
    """
    wrapper = PickyBPETokenizerWrapper(pickybpe_model_path)
    
    # Create tokenizer.json format
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "vocab": wrapper.vocab,
            "merges": [],
            "unk_token": wrapper.unk_token
        }
    }
    
    # Save temporary tokenizer.json
    tokenizer_path = Path("temp_tokenizer.json")
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    
    # Load as PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token=wrapper.unk_token,
        pad_token=wrapper.pad_token,
        cls_token=wrapper.cls_token,
        sep_token=wrapper.sep_token,
        mask_token=wrapper.mask_token,
    )
    
    return tokenizer


def prepare_dataset_from_tokenized_file(tokenized_file: str, max_length: int = 512):
    """
    Load pre-tokenized text file and prepare for BERT training
    
    Expected format: Each line contains space-separated token IDs
    Example: "45 123 67 89 12 ..."
    """
    dataset = load_dataset(
        'text',
        data_files={'train': tokenized_file},
        split='train'
    )
    
    def tokenize_function(examples):
        # Convert string of token IDs to list of integers
        input_ids = []
        for line in examples['text']:
            ids = [int(x) for x in line.strip().split()]
            # Truncate to max_length
            if len(ids) > max_length:
                ids = ids[:max_length]
            input_ids.append(ids)
        
        return {'input_ids': input_ids}
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Converting to token IDs"
    )
    
    return tokenized_dataset


def train_bert_base(
    pickybpe_model_path: str,
    tokenized_file: str,
    output_dir: str = './bert_output',
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    mlm_probability: float = 0.15,
    save_steps: int = 10000,
    logging_steps: int = 500
):
    """
    Train BERT base model from scratch on pre-tokenized corpus
    """
    
    # 1. Create tokenizer from PickyBPE model
    logger.info("Creating tokenizer from PickyBPE model...")
    tokenizer = create_tokenizer_from_pickybpe(pickybpe_model_path)
    
    # 2. Load and prepare dataset
    logger.info("Loading tokenized dataset...")
    train_dataset = prepare_dataset_from_tokenized_file(tokenized_file, max_length)
    
    # 3. Initialize BERT config
    logger.info("Initializing BERT configuration...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    
    # 4. Initialize BERT model
    logger.info("Initializing BERT model from scratch...")
    model = BertForMaskedLM(config)
    logger.info(f"Model has {model.num_parameters():,} parameters")
    
    # 5. Data collator for MLM (Masked Language Modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    
    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=10000,
        logging_dir=f'{output_dir}/logs',
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 = 32
    )
    
    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 8. Train!
    logger.info("Starting training...")
    trainer.train()
    
    # 9. Save final model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BERT with PickyBPE tokenizer')
    parser.add_argument('--pickybpe_model', type=str, required=True,
                        help='Path to PickyBPE model.json')
    parser.add_argument('--tokenized_file', type=str, required=True,
                        help='Path to pre-tokenized text file (space-separated token IDs)')
    parser.add_argument('--output_dir', type=str, default='./bert_output',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    train_bert_base(
        pickybpe_model_path=args.pickybpe_model,
        tokenized_file=args.tokenized_file,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    # CUDA_VISIBLE_DEVICES=6 python3 train_bert_peaky_bpe.py     --pickybpe_model ../langs_17_full_dataset_native_script/tok/picky_bpe_native_128k.json     --tokenized_file ../langs_17_full_dataset_native_script/dataset/sample_segment.txt     --output_dir ./output_sample     --epochs 2     --batch_size 8     --learning_rate 5e-5     --max_length 512 2>&1 | tee sample_debug.log