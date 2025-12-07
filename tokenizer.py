import sentencepiece as spm
from datasets import load_dataset

def train_sentencepiece_from_huggingface():
    # 1. Load Dataset in Streaming Mode
    # 'streaming=True' is crucial. It downloads data on the fly, 
    # so you don't need 50GB of RAM or disk space.
    print("Setting up stream...")
    dataset = load_dataset("ncduy/mt-en-vi", split="train", streaming=True)
    
    # 2. Define a Python Generator
    # SentencePiece needs an iterator that yields strings.
    # We loop through the dataset and yield the 'text' column.
    def batch_iterator(dataset_stream, limit=2800000):
        count = 0
        for i, item in enumerate(dataset_stream):
            if count >= limit:
                break
            
            # Extract the text content
            text = item.get("en", "")
            
            # Basic cleaning (optional but recommended)
            # Remove newlines to prevent sentence splitting issues if needed
            text = text.replace("\n", " ") 
            
            if text.strip():
                yield text
                count += 1
                
        print(f"Processed {count} sentences for training.")

    # 3. Train SentencePiece using 'sentence_iterator'
    # Note: We use 'sentence_iterator' instead of 'input'
    print("Starting training (this might take a while)...")
    
    # Create the iterator
    data_iter = batch_iterator(dataset, limit=2000000) # Adjust limit as needed
    
    spm.SentencePieceTrainer.train(
        sentence_iterator=data_iter,
        model_prefix='english_toknizer_spm', # Output filename (.model)
        vocab_size=32000,                    # As discussed for IWSLT
        character_coverage=0.9995,          # Good for Vietnamese
        model_type='unigram',               # Best for Translation
        pad_id=3,                           # Crucial for PyTorch
        unk_id=0,
        bos_id=1,
        eos_id=2,
        input_sentence_size=2000000,        # Buffer size for shuffling
        shuffle_input_sentence=True         # Randomize samples for better training
    )
    
    print("Training finished! Saved 'vietnamese_wiki_spm.model'")

if __name__ == "__main__":
    train_sentencepiece_from_huggingface()