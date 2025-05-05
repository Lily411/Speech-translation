from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq, AutomaticSpeechRecognitionPipeline
from peft import PeftConfig, PeftModel
import torch

language = "Korean"
language_decode = 'korean'
task = "transcribe"
model_name_or_path = "openai/whisper-large-v2"
model_dir = r'C:\Users\Lily\Desktop\python\llama\speech-to-text\ko\20250330\checkpoint-10000'

# 从预训练模型加载特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
# 从预训练模型加载分词器，可以指定语言和任务以获得最适合特定需求的分词器配置
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
# 从预训练模型加载处理器，处理器通常结合了特征提取器和分词器，为特定任务提供一站式的数据预处理
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

peft_config = PeftConfig.from_pretrained(model_dir)
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path,torch_dtype=torch.float16,device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, model_dir)

base_pipeline = AutomaticSpeechRecognitionPipeline(model = model, tokenizer = tokenizer, feature_extractor = feature_extractor)
peft_pipeline = AutomaticSpeechRecognitionPipeline(model = peft_model, tokenizer = tokenizer, feature_extractor = feature_extractor)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_decode, task=task)
#----------------------------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset, DatasetDict
from datasets import Audio

common_voice = DatasetDict()
common_voice["train"] = load_dataset(path='Bingsu/zeroth-korean', split="train")

#由48000轉做16000
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

small_common_voice = DatasetDict()
small_common_voice= common_voice["train"].shuffle(seed=411).select(range(12000, 13000))
small_common_voice = small_common_voice.rename_column('text', 'sentence')

#---------------------------------------------------------------------------------------------------------------------------------------
import torch
from evaluate import load
wer_metric = load("wer")

base_sum_wer = []
peft_sum_wer = []

for i in range(len(small_common_voice)):
  correct = small_common_voice["sentence"][i]

  try:
    with torch.cuda.amp.autocast():
      # 嘗試讀取音頻並進行語音識別
      base_text = base_pipeline(small_common_voice["audio"][i], max_new_tokens=255)["text"]
      peft_text = peft_pipeline(small_common_voice["audio"][i], max_new_tokens=255)["text"]
  except Exception as e:  # 捕獲所有異常，包括 soundfile.LibsndfileError
    print(f"處理第 {i} 個音頻時出錯: {e}")
    wer = 1  # 如果出錯，WER 設為 1.0（最差情況）
    continue  # 跳過當前文件，繼續下一個

  # 計算 WER
  base_wer = wer_metric.compute(references=[correct], predictions=[base_text])
  peft_wer = wer_metric.compute(references=[correct], predictions=[peft_text])

  print(f"i: {i} | base wer: {base_wer} | fine tune wer: {peft_wer}")
  print(f"base model predict: {base_text}")
  print(f"fine tune model predict: {peft_text}")
  print(f"correct: {correct}\n")

  # 確保 WER 在合理範圍（0~1）
  if peft_wer <= 1:
    base_sum_wer.append(base_wer)
    peft_sum_wer.append(peft_wer)
  else:
    base_sum_wer.append(1.0)
    peft_sum_wer.append(1.0)  # 如果 WER > 1，設為 1.0

base_mean_wer = sum(base_sum_wer) / len(base_sum_wer)
peft_mean_wer = sum(peft_sum_wer) / len(peft_sum_wer)

print(f"Base: the mean wer of korean speech to text is {base_mean_wer}")
print(f"Fine tune: the mean wer of korean speech to text is {peft_mean_wer}")
