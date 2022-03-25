# coding: utf-8

import t5
import seqio
import functools
import sklearn.metrics
from t5.data import TextLineTask
from t5.evaluation import metrics
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from sklearn.metrics import f1_score, precision_score, recall_score


DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)

DEFAULT_OUTPUT_FEATURES = {
	"inputs": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
	"targets": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

clef_multilingual_tsv_path = {
	"train":        "gs://nlp_base/mingzhe/clef/clef_multilingual/train.tsv",
	"dev":          "gs://nlp_base/mingzhe/clef/clef_multilingual/dev.tsv",
	"test":         "gs://nlp_base/mingzhe/clef/clef_multilingual/test.tsv",
}

def clef_metric(targets, predictions):
	metric_dict = {
		"precision": precision_score(targets, predictions, pos_label="yes", average='binary'),
		"recall": recall_score(targets, predictions, pos_label="yes", average='binary'),
		"f1": f1_score(targets, predictions, pos_label="yes", average='binary')
	}
	return metric_dict

def clef_dataset_fn(split, shuffle_files=False, lang="multilingual"):
	del shuffle_files
	print(f"Current Dataset [{lang}]")
	ds = tf.data.TextLineDataset(clef_multilingual_tsv_path[split])
	ds = ds.map(functools.partial(tf.io.decode_csv, record_defaults=["", ""], field_delim="\t", use_quote_delim=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
	return ds

def clef_preprocessor(ds):
	def normalize_text(text):
		text = tf.strings.lower(text)
		text = tf.strings.strip(text)
		text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
		return text

	def to_inputs_and_targets(ex):
		"""Map {"input": ..., "target": ...}->{"inputs": ..., "targets": ...}."""
		return {
			"inputs": normalize_text(ex["input"]),
			"targets": normalize_text(ex["target"])
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

seqio.TaskRegistry.add(
	"clef_multilingual",
	source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(clef_dataset_fn, lang="multilingual"),
		splits=["train", "dev"]
	),

	preprocessors=[
		clef_preprocessor,
		seqio.preprocessors.tokenize_and_append_eos,
	],

	postprocess_fn=t5.data.postprocessors.lower_text,
	metric_fns=[clef_metric],
	output_features=DEFAULT_OUTPUT_FEATURES,
)

seqio.MixtureRegistry.remove("clef_all")
seqio.MixtureRegistry.add(
	"clef_all",
	["clef_multilingual"], 
	default_rate=1.0
)