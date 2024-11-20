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
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer, DataCollatorWithPadding, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import os
import evaluate
from typing import Any, Dict, List, Union
# target = 'hanlo'
target = 'tailo'
# target_column = 'hok_text_hanlo_tai'
target_column = 'hok_text_tailo_number_tone'
size = 'small' # model size
data_source = 'tat_open_source'
# data_source = 'suisiann'
```


```python
dataset = load_dataset('csv', data_files={'train': pwd + '/data/' + data_source + '/dev/dev.tsv',
                                          'test': pwd + '/data/' + data_source + '/test/test.tsv'},
                       delimiter='\t', usecols=['hok_audio', target_column])
def update_audio_path(example, dataset_type):
    # Append the correct directory path based on the dataset type
    if dataset_type == 'train':
        example['hok_audio'] = pwd + '/data/' + data_source + f'/dev/{example["hok_audio"]}'
    elif dataset_type == 'test':
        example['hok_audio'] = pwd + '/data/' + data_source + f'/test/{example["hok_audio"]}'
    return example

# Apply the function to update paths for both train and test datasets
dataset['train'] = dataset['train'].map(lambda x: update_audio_path(x, 'train'))
dataset['test'] = dataset['test'].map(lambda x: update_audio_path(x, 'test'))
print(dataset['train'][0])

max_label_length = 448
def truncate_labels(example):
  """Truncates the 'labels' field to the maximum allowed length."""
  example[target_column] = example[target_column][:max_label_length]
  return example

# Apply the truncation function to your dataset
dataset = dataset.map(truncate_labels)
```

    {'hok_audio': './data/tat_open_source/dev/hok/TAT-Vol1-eval_0009_0_TAM0013_concat.wav', 'hok_text_tailo_number_tone': 'the5-si7 kha2 pian1-ho7:TA_0009'}



```python
feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-small', language='Mandarin', task='transcribe')
```


```python
input_str = dataset['train'][0][target_column]
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
processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='Mandarin', task='transcribe')
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

dataset = dataset.map(prepare_dataset, remove_columns=['hok_audio'])
```


```python
# Load the pre-trained Whisper model
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
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
text = dataset['train'][2][target_column]
tailo_tokenizer = TailoTokenizer()
tailo_tokens_split = tailo_tokenizer.tokenize(text)
tailo_tokens_string = tailo_tokenizer.tokenize_join(text)
tailo_tokens_string_no_dashes = tailo_tokenizer.tokenize_join_no_dashes(text)
print(text)
print(tailo_tokens_split)
print(tailo_tokens_string)
print(tailo_tokens_string_no_dashes)
```

    sua3-loh8-lai5 khuann3 lam5-tau5-kuan7 bin5-a2-tsai3 sann1 ho7 e5 thinn1-khi3
    ['s', 'ua3', '-', 'l', 'oh8', '-', 'l', 'ai5', 'kh', 'uann3', 'l', 'am5', '-', 't', 'au5', '-', 'k', 'uan7', 'b', 'in5', '-', 'a2', '-', 'ts', 'ai3', 's', 'ann1', 'h', 'o7', 'e5', 'th', 'inn1', '-', 'kh', 'i3']
    s ua3 - l oh8 - l ai5 kh uann3 l am5 - t au5 - k uan7 b in5 - a2 - ts ai3 s ann1 h o7 e5 th inn1 - kh i3
    s ua3 l oh8 l ai5 kh uann3 l am5 t au5 k uan7 b in5 a2 ts ai3 s ann1 h o7 e5 th inn1 kh i3



```python
# metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    if target == 'hanlo':
      metric = evaluate.load('cer')
      metric_name = "cer"
      # we do not want to group tokens when computing the metrics
      pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
      label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    else: # target == 'tailo'
      metric = evaluate.load('wer') # Reuse WER logic for token-level error rate
      metric_name = "ser"
      # decode predictions and labels using TailoTokenizer
      pred_str = [tailo_tokenizer.tokenize_join(p) for p in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
      label_str = [tailo_tokenizer.tokenize_join(l) for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

      # decode predictions and labels using TailoTokenizer without dashes "-" and "--"
      # pred_str = [tailo_tokenizer.tokenize_no_dashes(p) for p in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
      # label_str = [tokenize_sentence.tokenize_no_dashes(l) for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    metrics = 100 * metric.compute(predictions=pred_str, references=label_str)

    # print a few examples
    for i in range(min(5, len(pred_str))):  # Print first 5 examples
        print(f"Prediction: {pred_str[i]}")
        print(f"Ground Truth: {label_str[i]}")
        print("---")

    return {metric_name: metrics}
```


```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/"+ target + "-whisper-"+ size +"-training-logs",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=20,  # originally was 500
    # max_steps=100,  # originally was 5000
    num_train_epochs=5,  # Use epochs instead of max_steps
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
    metric_for_best_model="ser", # "cer"
    greater_is_better=False,
    push_to_hub=False,
)
```


```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
```

    /tmp/ipykernel_1709/2682642749.py:1: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer = Seq2SeqTrainer(



```python
trainer.train()
```

    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.
    `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...




    <div>

      <progress value='225' max='225' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [225/225 05:21, Epoch 4/5]
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


    /opt/conda/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 448, 'suppress_tokens': [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362], 'begin_suppress_tokens': [220, 50257]}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
      warnings.warn(





    TrainOutput(global_step=225, training_loss=0.7765229882134331, metrics={'train_runtime': 324.302, 'train_samples_per_second': 11.132, 'train_steps_per_second': 0.694, 'total_flos': 1.03198139154432e+18, 'train_loss': 0.7765229882134331, 'epoch': 4.945054945054945})




```python
save_path = pwd + '/model/' + target +'-whisper-'+ size +'-hokkien-finetuned'
print(save_path)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
```




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
  [86/86 03:02]
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


    Prediction: s uah4 l oh8 - l ai5 kh uann3 si in1 - t ik4 - tshi i7 b in5 - a2 - ts ai3 it4 - h o7 e5 th inn1 - kh i3 . 
    Ground Truth: s ua3 - l oh8 - l ai5 kh uann3 si in1 - t ik4 - tshi i7 b in5 - a2 - ts ai3 it4 h o7 e5 th inn1 - kh i3
    ---
    Prediction: un1 - t oo7 l i7 - ts ap8 - s ann1 t oo7 tsi i3 t i7 ts o7 - tshi it4 t oo7 , l oh8 - h oo7 g i7 - l ut8 l i7 - ts ap8 ph a1 ,  l ai5 - p in1 g oo7 - p ah4 g oo7 - ts ap8 k au3 h o7 tshi iann2 - l ai5 ts ap8 s ann1 h o7 k ui7 - t ai5 p ang1 - l i2 . 
    Ground Truth: un1 - t oo7 l i7 - ts ap8 - s ann1 t oo7 tsi i3 ji i7 - ts ap8 tshi it4 t oo7 , l oh8 - h oo7 k i1 - l ut8 l i7 ts ap8 %  l ai5 - p in1 g oo7 - p ah4 g oo7 - ts ap8 k au2 h o7 tshi iann2 - l ai5 ts ap8 - s ann1 h o7 k ui7 - t ai5 p an7 - l i2
    ---
    Prediction: l ong2 - ts ong2 p eh4 - p ah4 kh ong3 - ji i7 kh oo1 , l au7 l i2 k au2 - ts ap8 p eh4 kh oo1
    Ground Truth: l ong2 - ts ong2 p eh4 p ah4 kh ong3 - ji i7 kh oo1 , ts au7 l i2 k au2 - ts ap8 p eh4 kh oo1
    ---
    Prediction: k in1 - a2 - ji i7 si i7 s ann1 - g ueh8 ts ap8 - s ann1 , p ai3 - l ak8
    Ground Truth: k in1 - a2 - l it8 si i7 s ann1 - g ueh8 ts ap8 - s ann1 , p ai3 - l ak8
    ---
    Prediction: i1 - t i7 it4 k iu2 s u3 in1 n i5 tsh ut4 - si i3
    Ground Truth: i1 - t i7 it4 k iu2 s u3 it4 n i5 tsh ut4 - si i3
    ---
    {'eval_loss': 0.7358824014663696, 'eval_ser': 30.930688143288037, 'eval_runtime': 227.8439, 'eval_samples_per_second': 3.011, 'eval_steps_per_second': 0.377, 'epoch': 4.945054945054945}



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


    Transcription: {'text': 'i1-ki1 leh4 tiam2 tsiu2 tsit4-pue3-tsit4-pue3-tsit4-pue3 e5 ta1tshiau3 li7 ai3 ti7-liong7-gua2gua2 tsiu2-liong7 bo5-hoo7 mai7-ka1 gua2 tshong3-khang1si7 kang1-tsit4-kang1,tsit4-kang1,tsit4-kang1 e5 tsa1-hu7kua2 tsit4-tsit4-tsit4 e5 lau5.', 'chunks': [{'timestamp': (0.0, 6.32), 'text': 'i1-ki1 leh4 tiam2 tsiu2 tsit4-pue3-tsit4-pue3-tsit4-pue3 e5 ta1'}, {'timestamp': (6.32, 10.52), 'text': 'tshiau3 li7 ai3 ti7-liong7-gua2'}, {'timestamp': (10.52, 14.28), 'text': 'gua2 tsiu2-liong7 bo5-hoo7 mai7-ka1 gua2 tshong3-khang1'}, {'timestamp': (14.28, 18.32), 'text': 'si7 kang1-tsit4-kang1,tsit4-kang1,tsit4-kang1 e5 tsa1-hu7'}, {'timestamp': (18.32, 21.32), 'text': 'kua2 tsit4-tsit4-tsit4 e5 lau5.'}]}


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
