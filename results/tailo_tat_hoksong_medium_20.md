```python
pwd = '.'
# !pip install --upgrade pip
# !pip install --upgrade datasets transformers accelerate evaluate jiwer
# from google.colab import drive
# drive.mount('/content/drive')
# pwd = './drive/MyDrive/Colab Notebooks/CS4347'
```


```python
import torch
import torchaudio
import tensorboard
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer, DataCollatorWithPadding, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import os
import evaluate
from typing import Any, Dict, List, Union
# target = 'hanlo'
target = 'tailo'
# target_column = 'hok_text_hanlo_tai'
target_column = 'hok_text_tailo_number_tone'
size = 'medium' # model size
n_epoch = 20
```


```python
from datasets import load_dataset, DatasetDict

# Specify datasets and their split status
data_sources = [
    {"name": "tat_open_source", "pre_split": True},
    {"name": "hok_song", "pre_split": False, "test_split_percentage": 1},
    # {"name": "suisiann", "pre_split": False, "data_percentage": 0.1, "test_split_percentage": 0.25},
]

# Initialize an empty DatasetDict
combined_dataset = DatasetDict()

# Loop through each dataset
for data_source in data_sources:
    dataset_name = data_source["name"]
    is_pre_split = data_source["pre_split"]
    data_percentage = data_source.get("data_percentage", 1.0)  # Default to 100% if not specified
    test_split_percentage = data_source.get("test_split_percentage", 0.2) # Default to 20% if not specified
    
    if is_pre_split:
        # For pre-split datasets, load train and test directly
        dataset = load_dataset(
            'csv',
            data_files={
                'train': pwd + f'/data/{dataset_name}/dev/dev.tsv',
                'test': pwd + f'/data/{dataset_name}/test/test.tsv'
            },
            delimiter='\t',
            usecols=['hok_audio', target_column]
        )
    else:
        # Load the non-pre-split dataset
        dataset = load_dataset(
            'csv',
            data_files={'full': pwd + f'/data/{dataset_name}/all.csv'}
        )
    
        # Filter columns using map
        dataset = dataset['full'].map(lambda example: {key: example[key] for key in ['hok_audio', target_column]})
    
        # Dynamically split into train and test
        dataset = dataset.train_test_split(test_size=test_split_percentage)

    # Apply data percentage (limit the rows based on the percentage)
    if data_percentage < 1.0:
        dataset['train'] = dataset['train'].select(range(int(len(dataset['train']) * data_percentage)))
        dataset['test'] = dataset['test'].select(range(int(len(dataset['test']) * data_percentage)))

    def update_audio_path(example, dataset_type):
        if is_pre_split:
            if dataset_type == 'train':
                example['hok_audio'] = pwd + f'/data/{dataset_name}/dev/' + example['hok_audio']
            elif dataset_type == 'test':
                example['hok_audio'] = pwd + f'/data/{dataset_name}/test/' + example['hok_audio']
        else:
            example['hok_audio'] = pwd + f'/data/{dataset_name}/' + example['hok_audio']
        return example

    dataset['train'] = dataset['train'].map(lambda x: update_audio_path(x, 'train'))
    dataset['test'] = dataset['test'].map(lambda x: update_audio_path(x, 'test'))

    # Add a `source` column to indicate the dataset name
    dataset['train'] = dataset['train'].map(lambda x: {**x, 'source': dataset_name})
    dataset['test'] = dataset['test'].map(lambda x: {**x, 'source': dataset_name})

    # Add the current dataset's splits to the combined dataset
    if 'train' not in combined_dataset:
        combined_dataset['train'] = dataset['train']
    else:
        combined_dataset['train'] = concatenate_datasets([combined_dataset['train'], dataset['train']])
    
    if 'test' not in combined_dataset:
        combined_dataset['test'] = dataset['test']
    else:
        combined_dataset['test'] = concatenate_datasets([combined_dataset['test'], dataset['test']])

# Truncate labels for the combined dataset
max_label_length = 448

def truncate_labels(example):
    """Truncates the 'labels' field to the maximum allowed length."""
    example[target_column] = example[target_column][:max_label_length]
    return example

combined_dataset['train'] = combined_dataset['train'].map(truncate_labels)
combined_dataset['test'] = combined_dataset['test'].map(truncate_labels)
```


    Map:   0%|          | 0/14 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1 [00:00<?, ? examples/s]



    Map:   0%|          | 0/14 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1 [00:00<?, ? examples/s]



    Map:   0%|          | 0/736 [00:00<?, ? examples/s]



    Map:   0%|          | 0/687 [00:00<?, ? examples/s]



```python
# test dataset loading
print(combined_dataset['train'].num_rows)
print(combined_dataset['train'][710])
print(combined_dataset['train'][730])
```

    736
    {'hok_audio': './data/tat_open_source/dev/hok/TAT-Vol1-eval_0034_5.64_TSM013_concat.wav', 'hok_text_tailo_number_tone': 'hian7-tai7 e5 tai5-uan5 siau3-lian5-lang5 lian5“bong1 la5-a2”to1 m7 tsai1 siann2 i3-su3, beh4 an2-tsuann2 ka7 kai2-sueh4“kiam1 se2 khoo3”?', 'source': 'tat_open_source'}
    {'hok_audio': './data/hok_song/4-1.wav', 'hok_text_tailo_number_tone': 'Pue1 ti7 hong1 tiong1 e5 sio2 hoo7 Ia7-ia7 ti7 gua2 e5 thang1 tsing5 Uan3 tsit1-tiau5 ai2-tsing5-loo7 Uan3 tsit1-tiau5 ai3-tsing5-loo7 Kam2-si7 oo1-hun5 khi2 uan3-too3 Tsiong1 gun2 gueh8-niu5 e5 bak8-sai2 Tshue1 ling2-ling2 e5 hong1', 'source': 'hok_song'}



```python
feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-' + size)
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-' + size, language='Mandarin', task='transcribe')
```


```python
input_str = combined_dataset['train'][0][target_column]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
```


```python
# test
print(input_str)
print(labels)
print(decoded_with_special)
print(decoded_str)
input_str == decoded_str
```

    the5-si7 kha2 pian1-ho7:TA_0009
    [50258, 50260, 50359, 50363, 3322, 20, 12, 7691, 22, 350, 1641, 17, 32198, 16, 12, 1289, 22, 25, 8241, 62, 1360, 24, 50257]
    <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>the5-si7 kha2 pian1-ho7:TA_0009<|endoftext|>
    the5-si7 kha2 pian1-ho7:TA_0009





    True




```python
processor = WhisperProcessor.from_pretrained('openai/whisper-' + size, language='Mandarin', task='transcribe')
```


```python
def preprocess_function(examples):
    audio_path = examples['hok_audio']
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    # Resample if necessary
    speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
    # Convert audio to log-mel spectrogram
    input_features = processor(speech_array.squeeze().numpy(), sampling_rate=16000).input_features
    return {'input_features': input_features, 'transcription': examples[target_column]}

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio_path = batch['hok_audio']
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)

    speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
    # compute log-Mel input features from input audio array
    batch["input_features"] =  feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000).input_features[0]
    # batch["input_features"] = feature_extractor(speech_array, sampling_rate=16000).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch[target_column]).input_ids
    return batch

combined_dataset = combined_dataset.map(prepare_dataset, remove_columns=['hok_audio'])
```


    Map:   0%|          | 0/736 [00:00<?, ? examples/s]



    Map:   0%|          | 0/687 [00:00<?, ? examples/s]



```python
# Load the pre-trained Whisper model
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-' + size)
```


```python
model.generation_config.language = 'Mandarin'
model.generation_config.task = 'transcribe'

model.generation_config.forced_decoder_ids = None
```


```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```


```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
```


```python
# Tailo Tokenizer
#   code snippet from https://github.com/wchang88/Tai-Lo-Tokenizer/blob/main/TailoTokenizer.py
import re
from string import punctuation

class TailoTokenizer():
   def __init__(self):
      self.consonants = ['ph', 'p',
                      'm', 'b',
                      'tshi', 'tsh', 'tsi', 'ts', 'th','t',
                      'n', 'l',
                      'kh', 'k',
                      'ng', 'g',
                      'si', 's',
                      'ji','j',
                      'h']

   def tokenize_helper(self, word):
      for onset in self.consonants:
         if word.lower().find(onset) == 0:
            if onset[-1] == 'i':
               return [word[:len(onset)], word[len(onset) - 1:]]
            else:
               return [word[:len(onset)], word[len(onset):]]
      return [word]

   def tokenize(self, sent):
      tokens = []
      for word in re.split(r' |([%s]+)' % re.escape(punctuation), sent):
         if word is not None:
            if re.search(r'[%s]+' % re.escape(punctuation), word):
               # if any combination of punctuation
               tokens.append(word)
            else:
               # if a tai-lo romanization
               tokens.extend(self.tokenize_helper(word))
      return tokens

   def tokenize_join(self, text):
      # Tokenize into initials and finals
      tokens = self.tokenize(text)
      # Join tokens with spaces for consistency
      return " ".join(tokens)

   def tokenize_join_no_dashes(self, text): # remove "--"" and "-"" in Tailo (not used)
      # Remove dashes between words
      text = text.replace("--", " ").replace("-", " ")
      # Tokenize into initials and finals
      tokens = self.tokenize(text)
      # Join tokens with spaces for consistency
      return " ".join(tokens)

   def remove_tone_numbers(self, token):
      """Removes trailing tone numbers from a token."""
      return re.sub(r'\d+$', '', token)

   def tokenize_join_remove_tones(self, text):
      tokens = self.tokenize(text)
      tokens = [self.remove_tone_numbers(token) for token in tokens]
      return " ".join(tokens)

   def tokenize_join_no_dashes_remove_tones(self, text):
      text = text.replace("--", " ").replace("-", " ")
      tokens = self.tokenize(text)
      tokens = [self.remove_tone_numbers(token) for token in tokens]
      return " ".join(tokens)

   def detokenize(self, tokens):
      i = 0
      sentence = []
      dash_found = False
      while i < len(tokens):
         if re.search(r'[%s]+' % re.escape(punctuation), tokens[i]):
            # if the current token is punctuation
            if '-' in tokens[i]:
               dash_found = True
            sentence.append(tokens[i])
            i += 1
         else:
            if tokens[i] in self.consonants:
               # if the current token is a consonant, combine it with the next
               if tokens[i][-1] == 'i' and tokens[i+1][0] == 'i':
                  # reduce double i into single i
                  sentence.append("".join([tokens[i], tokens[i+1][1:]]))
               else:
                  sentence.append("".join(tokens[i:i+2]))
               i += 2
            else:
               sentence.append(tokens[i])
               i += 1

            if dash_found:
               compound = [sentence.pop() for i in range(3)]
               sentence.append("".join(compound[::-1]))
               dash_found = False

      return " ".join(sentence)
```


```python
# test Tailo Tokenizer
text = combined_dataset['train'][2][target_column]
tailo_tokenizer = TailoTokenizer()
tailo_tokens_split = tailo_tokenizer.tokenize(text)
tailo_tokens_string = tailo_tokenizer.tokenize_join(text)
tailo_tokens_string_no_dashes = tailo_tokenizer.tokenize_join_no_dashes(text)

tailo_tokens_string_no_tones = tailo_tokenizer.tokenize_join_remove_tones(text)
tailo_tokens_string_no_dashes_no_tones = tailo_tokenizer.tokenize_join_no_dashes_remove_tones(text)
print(text)
print(tailo_tokens_split)
print(tailo_tokens_string)
print(tailo_tokens_string_no_dashes)
print(tailo_tokens_string_no_tones)
print(tailo_tokens_string_no_dashes_no_tones)
```

    sua3-loh8-lai5 khuann3 lam5-tau5-kuan7 bin5-a2-tsai3 sann1 ho7 e5 thinn1-khi3
    ['s', 'ua3', '-', 'l', 'oh8', '-', 'l', 'ai5', 'kh', 'uann3', 'l', 'am5', '-', 't', 'au5', '-', 'k', 'uan7', 'b', 'in5', '-', 'a2', '-', 'ts', 'ai3', 's', 'ann1', 'h', 'o7', 'e5', 'th', 'inn1', '-', 'kh', 'i3']
    s ua3 - l oh8 - l ai5 kh uann3 l am5 - t au5 - k uan7 b in5 - a2 - ts ai3 s ann1 h o7 e5 th inn1 - kh i3
    s ua3 l oh8 l ai5 kh uann3 l am5 t au5 k uan7 b in5 a2 ts ai3 s ann1 h o7 e5 th inn1 kh i3
    s ua - l oh - l ai kh uann l am - t au - k uan b in - a - ts ai s ann h o e th inn - kh i
    s ua l oh l ai kh uann l am t au k uan b in a ts ai s ann h o e th inn kh i



```python
# metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str_raw = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str_raw = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Hanlo case: Use CER
    if target == 'hanlo':
        # Load CER metric
        cer_metric = evaluate.load('cer')

        # Calculate CER
        cer = cer_metric.compute(predictions=pred_str_raw, references=label_str_raw)

        # Print examples for debugging
        for i in range(min(5, len(pred_str_raw))):  # Print first 5 examples
            print(f"Prediction: {pred_str_raw[i]}")
            print(f"Ground Truth: {label_str_raw[i]}")
            print("---")

        return {
            "cer": 100 * cer  # CER as percentage
        }

    # Tailo case: Calculate multiple metrics
    else:
        # Initialize TailoTokenizer
        tailo_tokenizer = TailoTokenizer()

        # Processed strings for different metrics
        pred_str_tokenize = [tailo_tokenizer.tokenize_join(p) for p in pred_str_raw]
        label_str_tokenize = [tailo_tokenizer.tokenize_join(l) for l in label_str_raw]

        pred_str_no_tones = [tailo_tokenizer.tokenize_join_remove_tones(p) for p in pred_str_raw]
        label_str_no_tones = [tailo_tokenizer.tokenize_join_remove_tones(l) for l in label_str_raw]

        # Load WER metric
        wer_metric = evaluate.load('wer')

        # Calculate WER for raw text
        wer = wer_metric.compute(predictions=pred_str_raw, references=label_str_raw)

        # SER for tokenized text (after `tokenize_join`)
        ser = wer_metric.compute(predictions=pred_str_tokenize, references=label_str_tokenize)

        # SER for tokenized text with tones removed (after `tokenize_join_remove_tones`)
        ser_no_tones = wer_metric.compute(predictions=pred_str_no_tones, references=label_str_no_tones)

        # Print examples for debugging
        for i in range(min(5, len(pred_str_raw))):  # Print first 5 examples
            print(f"Original Prediction: {pred_str_raw[i]}")
            print(f"Original Ground Truth: {label_str_raw[i]}")
            print(f"Tokenized Prediction: {pred_str_tokenize[i]}")
            print(f"Tokenized Ground Truth: {label_str_tokenize[i]}")
            print(f"Prediction without Tones: {pred_str_no_tones[i]}")
            print(f"Ground Truth without Tones: {label_str_no_tones[i]}")
            print("---")

        # Return all metrics
        return {
            "wer": 100 * wer,  # Original WER
            "ser": 100 * ser,  # SER after `tokenize_join`
            "ser_no_tones": 100 * ser_no_tones  # SER after `tokenize_join_remove_tones`
        }
```


```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/"+ target + "-whisper-"+ size +"-training-logs",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=20,  # originally was 500
    # max_steps=100,  # originally was 5000
    num_train_epochs=n_epoch,  # Use epochs instead of max_steps
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer" if target == "hanlo" else "ser", 
    greater_is_better=False,
    push_to_hub=False,
)
```


```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=combined_dataset['train'],
    eval_dataset=combined_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
```

    /tmp/ipykernel_4647/1919396851.py:1: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer = Seq2SeqTrainer(



```python
trainer.train()
```

    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
    `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...




    <div>

      <progress value='920' max='920' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [920/920 56:13, Epoch 20/20]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>


    /opt/conda/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 448, 'suppress_tokens': [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362], 'begin_suppress_tokens': [220, 50257]}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
      warnings.warn(





    TrainOutput(global_step=920, training_loss=0.12274615124586727, metrics={'train_runtime': 3377.6897, 'train_samples_per_second': 4.358, 'train_steps_per_second': 0.272, 'total_flos': 1.50233042386944e+19, 'train_loss': 0.12274615124586727, 'epoch': 20.0})




```python
save_path = pwd + '/model/' + target +'-whisper-'+ size +'-hokkien-finetuned-' + str(n_epoch)
print(save_path)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
```

    ./model/tailo-whisper-medium-hokkien-finetuned-20





    []




```python
# Evaluate
results = trainer.evaluate()
print(results)
```

    You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, 50259], [2, 50359], [3, 50363]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.




<div>

  <progress value='86' max='86' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [86/86 06:58]
</div>



    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
    Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.


    Original Prediction: sua3-loh8-lai5 khuann3 sin1-tioh8-tshi7 bin5-a2-tsai3 it4 ho7 e5 thinn1-khi3
    Original Ground Truth: sua3-loh8-lai5 khuann3 sin1-tik4-tshi7 bin5-a2-tsai3 it4 ho7 e5 thinn1-khi3
    Tokenized Prediction: s ua3 - l oh8 - l ai5 kh uann3 si in1 - t ioh8 - tshi i7 b in5 - a2 - ts ai3 it4 h o7 e5 th inn1 - kh i3
    Tokenized Ground Truth: s ua3 - l oh8 - l ai5 kh uann3 si in1 - t ik4 - tshi i7 b in5 - a2 - ts ai3 it4 h o7 e5 th inn1 - kh i3
    Prediction without Tones: s ua - l oh - l ai kh uann si in - t ioh - tshi i b in - a - ts ai it h o e th inn - kh i
    Ground Truth without Tones: s ua - l oh - l ai kh uann si in - t ik - tshi i b in - a - ts ai it h o e th inn - kh i
    ---
    Original Prediction: un1-too7 ji7-tsap8-sann1 too7 tsi3 ti7-tsap8 tshit4 too7,loh8-hoo7 ki1-lut8 ji7-tsap8 pha1 lai5-pin1 goo7-pah4 goo7-tsap8 kau3 ho7 tshiann2-lai5 tsap8-sann1 ho7 kui7-tai5 pan7-li2
    Original Ground Truth: un1-too7 li7-tsap8-sann1 too7 tsi3 ji7-tsap8 tshit4 too7,loh8-hoo7 ki1-lut8 li7 tsap8% lai5-pin1 goo7-pah4 goo7-tsap8 kau2 ho7 tshiann2-lai5 tsap8-sann1 ho7 kui7-tai5 pan7-li2
    Tokenized Prediction: un1 - t oo7 ji i7 - ts ap8 - s ann1 t oo7 tsi i3 t i7 - ts ap8 tshi it4 t oo7 , l oh8 - h oo7 k i1 - l ut8 ji i7 - ts ap8 ph a1 l ai5 - p in1 g oo7 - p ah4 g oo7 - ts ap8 k au3 h o7 tshi iann2 - l ai5 ts ap8 - s ann1 h o7 k ui7 - t ai5 p an7 - l i2
    Tokenized Ground Truth: un1 - t oo7 l i7 - ts ap8 - s ann1 t oo7 tsi i3 ji i7 - ts ap8 tshi it4 t oo7 , l oh8 - h oo7 k i1 - l ut8 l i7 ts ap8 %  l ai5 - p in1 g oo7 - p ah4 g oo7 - ts ap8 k au2 h o7 tshi iann2 - l ai5 ts ap8 - s ann1 h o7 k ui7 - t ai5 p an7 - l i2
    Prediction without Tones: un - t oo ji i - ts ap - s ann t oo tsi i t i - ts ap tshi it t oo , l oh - h oo k i - l ut ji i - ts ap ph a l ai - p in g oo - p ah g oo - ts ap k au h o tshi iann - l ai ts ap - s ann h o k ui - t ai p an - l i
    Ground Truth without Tones: un - t oo l i - ts ap - s ann t oo tsi i ji i - ts ap tshi it t oo , l oh - h oo k i - l ut l i ts ap %  l ai - p in g oo - p ah g oo - ts ap k au h o tshi iann - l ai ts ap - s ann h o k ui - t ai p an - l i
    ---
    Original Prediction: long2-tsong2 peh4-pah4 khong3 ji7 khoo1,tsau7 li2 kau2-tsap8 peh4 khoo1
    Original Ground Truth: long2-tsong2 peh4 pah4 khong3-ji7 khoo1,tsau7 li2 kau2-tsap8 peh4 khoo1
    Tokenized Prediction: l ong2 - ts ong2 p eh4 - p ah4 kh ong3 ji i7 kh oo1 , ts au7 l i2 k au2 - ts ap8 p eh4 kh oo1
    Tokenized Ground Truth: l ong2 - ts ong2 p eh4 p ah4 kh ong3 - ji i7 kh oo1 , ts au7 l i2 k au2 - ts ap8 p eh4 kh oo1
    Prediction without Tones: l ong - ts ong p eh - p ah kh ong ji i kh oo , ts au l i k au - ts ap p eh kh oo
    Ground Truth without Tones: l ong - ts ong p eh p ah kh ong - ji i kh oo , ts au l i k au - ts ap p eh kh oo
    ---
    Original Prediction: kin1-a2-jit8 si7 sann1-gueh8 tsap8-sann1,pai3-lak8
    Original Ground Truth: kin1-a2-lit8 si7 sann1-gueh8 tsap8-sann1,pai3-lak8
    Tokenized Prediction: k in1 - a2 - ji it8 si i7 s ann1 - g ueh8 ts ap8 - s ann1 , p ai3 - l ak8
    Tokenized Ground Truth: k in1 - a2 - l it8 si i7 s ann1 - g ueh8 ts ap8 - s ann1 , p ai3 - l ak8
    Prediction without Tones: k in - a - ji it si i s ann - g ueh ts ap - s ann , p ai - l ak
    Ground Truth without Tones: k in - a - l it si i s ann - g ueh ts ap - s ann , p ai - l ak
    ---
    Original Prediction: i1-ti7 it4 kiu2 su3 in1-ni5 tshut4-si3
    Original Ground Truth: i1-ti7 it4 kiu2 su3 it4 ni5 tshut4-si3
    Tokenized Prediction: i1 - t i7 it4 k iu2 s u3 in1 - n i5 tsh ut4 - si i3
    Tokenized Ground Truth: i1 - t i7 it4 k iu2 s u3 it4 n i5 tsh ut4 - si i3
    Prediction without Tones: i - t i it k iu s u in - n i tsh ut - si i
    Ground Truth without Tones: i - t i it k iu s u it n i tsh ut - si i
    ---
    {'eval_loss': 1.0137630701065063, 'eval_wer': 60.20255495454022, 'eval_ser': 23.494796828543112, 'eval_ser_no_tones': 16.927325761807236, 'eval_runtime': 469.3256, 'eval_samples_per_second': 1.464, 'eval_steps_per_second': 0.183, 'epoch': 20.0}



```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

asr_model = WhisperForConditionalGeneration.from_pretrained(save_path)
processor = WhisperProcessor.from_pretrained(save_path)

asr_pipeline = pipeline("automatic-speech-recognition",
                        model=asr_model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        chunk_length_s=30,
                        batch_size=16,  # batch size for inference - set based on your device
                        torch_dtype=torch_dtype,
                        device=device)
```


```python
test_file_name = '/test_hokkien.mp3'
test_audio_path = pwd + test_file_name
# Perform inference on a new audio file
transcription = asr_pipeline(test_audio_path, return_timestamps=True)
print(f"Transcription: {transcription}")
```

    /opt/conda/lib/python3.11/site-packages/transformers/models/whisper/generation_whisper.py:509: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
      warnings.warn(


    Transcription: {'text': 'tsit8-ki1 le5 tiam3, tsiu7 tsit8-pue1 tsit8-pue1 tsit8-pue1 le5 tah4, tshiann2 li2 ai3 the5-liong7 gua2, gua2 tsiu7-liong7 bo5 ho2 mai3 ka7 gua2 tshong3-khang1, si7 kan1 tsit8 kang1 tsit8 kang1 tsit8 kang1 le5 tsau5, kuann2 tsit8-ti7-tsit8-tsit8 le5 lau5', 'chunks': [{'timestamp': (0.0, 21.0), 'text': 'tsit8-ki1 le5 tiam3, tsiu7 tsit8-pue1 tsit8-pue1 tsit8-pue1 le5 tah4, tshiann2 li2 ai3 the5-liong7 gua2, gua2 tsiu7-liong7 bo5 ho2 mai3 ka7 gua2 tshong3-khang1, si7 kan1 tsit8 kang1 tsit8 kang1 tsit8 kang1 le5 tsau5, kuann2 tsit8-ti7-tsit8-tsit8 le5 lau5'}]}


薰一枝一枝一枝咧點
hun tsi̍t ki tsi̍t ki leh tiám

酒一杯一杯一杯咧焦
tsiú tsi̍t pue tsi̍t pue tsi̍t pue leh ta

請你愛體諒我
tshiánn lí ài thé-liōng guá

我酒量無好　莫共我創空
guá tsiú-liōng bô hó, mài kā guá tshòng-khang

時間一工一工一工咧走
sî-kan tsi̍t kang tsi̍t kang tsi̍t kang leh tsáu

汗一滴一滴一滴咧流
kuann tsi̍t tih tsi̍t tih tsi̍t tih leh lâu

有一工　咱攏老
ū tsi̍t kang, lán lóng lāu

𤆬某囝鬥陣
tshuā bóo-kiánn tàu-tīn

浪子回頭
lōng-tsú huê-thâu
