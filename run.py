import pdb
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor
from datasets import load_dataset
from jiwer import wer
from datasets import Audio
from transformers import Seq2SeqTrainer
# Load your manifest files
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

def compute_wer(predictions, references):
    return wer(references, predictions)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
    
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["transcription"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None


dataset = load_dataset("./", data_dir="./data")

sampling_rate = processor.feature_extractor.sampling_rate
dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-asr",
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    save_total_limit=2,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_first_step=True,
    remove_unused_columns=True,
)




trainer = Seq2SeqTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_wer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=feature_extractor,
)

trainer.train()