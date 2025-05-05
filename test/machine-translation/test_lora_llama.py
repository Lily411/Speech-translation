import sys

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, AutoPeftModelForCausalLM,LoraModel, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from datasets import Dataset
import jieba
from deep_translator import GoogleTranslator
#----------------------------------------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
  device = "cuda"
#----------------------------------------------------------------------------------------------
# 載入 llama 和訓練好lora 模型
model_id = 'meta-llama/Meta-Llama-3-8B'
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_id, token = token,trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    torch_dtype=torch.float16,
    device_map = device
)
lora_model = 'C:\\Users\\Lily\\Desktop\\python\\llama\\en_ko_ja_500\\checkpoint-500'

model = PeftModel.from_pretrained(base_model, lora_model)
model.to(device)
#----------------------------------------------------------------------------------------------
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#
# ### Instruction:
# {}
#
# ### Input:
# {}
#
# ### Response:
# {}"""
#
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "translate korean to chinese:", # instruction
#         "얼룩말 줄무늬는 각 개체마다 고유한 패턴이 있습니다. 이러한 패턴의 기능에 대한 여러 이론이 제안되었으며, 대부분의 증거는 파리에 대한 억제력으로 이를 뒷받침합니다. 얼룩말은 동부 및 남부 아프리카에 서식하며 사바나, 초원, 삼림, 관목지, 산악 지역과 같은 다양한 서식지에서 발견될 수 있습니다.", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda")
#
# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
# sys.exit(0)
#----------------------------------------------------------------------------------------------
# test_prompt = 'translate english to chinese:\n input:' + 'good morning' +'\noutput:'
#
# model_input = tokenizer(test_prompt, return_tensors = 'pt').to(device)
# model.eval()
# with torch.no_grad():
#     res = model.generate(**model_input, max_new_tokens = 100)[0]
#     res_decode = tokenizer.decode(res, skip_special_tokens=True)
#
#     output_start =  res_decode.index("output:") + len("output:")
#     output_end =  res_decode.index("</s>")
#     res_output = res_decode[output_start:output_end]
#     print(res_decode)
#     print(res_output)
#----------------------------------------------------------------------------------------------
# BLEU function

import  nltk
# from nltk.translate.bleu_score import  SmoothingFunction
# import jieba
#
# def sentence_to_ngrams(sentence, n):
#     words = jieba.lcut(sentence)
#     return set(nltk.ngrams(words, n))
#
# def calculate_bleu(reference, candidate, max_n = 4):
#     smooth = SmoothingFunction().method4
#     scores = []
#     for n in range(1, max_n + 1):
#         reference_ngrams = sentence_to_ngrams(reference, n)
#         candidate_ngrams = sentence_to_ngrams(candidate, n)
#         if reference_ngrams and candidate_ngrams:
#             scores.append(nltk.translate.bleu_score.sentence_bleu([reference_ngrams], candidate_ngrams,smoothing_function=smooth))
#
#     final_score = max(scores)
#     return final_score
# #----------------------------------------------------------------------------------------------
# 攞data
# dataset = load_dataset(path='hosannaho/enzh', split="train")
# dataset_eval = dataset.select(range(20000, 22000))

#----------------------------------------------------------------------------------------------
# 計算bleu score
# total_bleu = []
# for i in range(len(dataset_eval)):
#     en = dataset_eval['english'][i]
#     zh = dataset_eval['chinese'][i]
#
#     eval_prompt = 'translate english to chinese:\n input:' + en + '\noutput:'
#     model_input = tokenizer(eval_prompt, return_tensors='pt').to(device)
#     model.eval()
#     with torch.no_grad():
#         res = model.generate(**model_input, max_new_tokens=100)[0]
#         res_decode = tokenizer.decode(res, skip_special_tokens=True)
#         output_start = res_decode.index("output:") + len("output:")
#         if "。" in res_decode:
#             output_end = res_decode.index("。") + 1
#         else :
#             output_end = len(res_decode)
#         res_output = res_decode[output_start:output_end].replace("</s>", "")
#
#
#     reference = zh
#     candidate = res_output
#
#     bleu_scores = calculate_bleu(reference, candidate)
#     total_bleu.append(bleu_scores)
#
#     print(i)
#     print(f"英文原句子: {en}")
#     print(f"model 翻譯的句子: {res_output}")
#     print(f"正確句子翻譯的:    {zh}")
#     print(f"Bleu_scores: {bleu_scores}")
#     print()
#
# final_bleu = sum(total_bleu) / len(total_bleu)
# print(f"最終 Bleu_scores: {final_bleu}")
#----------------------------------------------------------------------------------------------
#　meteor
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
#----------------------------------------------------------------------------------------------
# en Meteor
dataset = load_dataset(path='hosannaho/enzh', split="train")
dataset_eval = dataset.select(range(5000, 5100))

translator = GoogleTranslator(source='en', target='zh-CN')

total_meteor = []
google_meteor = []
for i in range(len(dataset_eval)):
    en = dataset_eval['english'][i]
    zh = dataset_eval['chinese'][i]

    eval_prompt = 'translate english to chinese:\n input:' + en + '\noutput:'
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "translate english to chinese:",  # instruction
                en,# input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    de_output = tokenizer.batch_decode(outputs)
    res_output = de_output[0][de_output[0].index("Response:") + len("Response:\n"):de_output[0].index("<|end_of_text|>") ]
    google_output = translator.translate(en)

    reference = jieba.lcut(zh)
    candidate = jieba.lcut(res_output)
    google_candidate = jieba.lcut(google_output)

    meteor = meteor_score([reference], candidate)
    total_meteor.append(meteor)

    g_meteor = meteor_score([reference], google_candidate)
    google_meteor.append(g_meteor)

    print(i)
    print(f"英文原句子: {en}")
    print(f"model 翻譯的句子: {res_output}")
    print(f"google 翻譯的句子: {google_output}")
    print(f"正確句子翻譯的:    {zh}")
    print(f"model_scores: {meteor}")
    print(f"google_scores: {g_meteor}")
    print()

final_meteor = sum(total_meteor) / len(total_meteor)
print(f"最終 model meteor_scores: {final_meteor}")
final_google_meteor = sum(google_meteor) / len(google_meteor)
print(f"最終 google meteor_scores: {final_google_meteor}")
#----------------------------------------------------------------------------------------------
#ko Meteor
dataset = load_dataset(path='traintogpb/aihub-kozh-translation-integrated-base-1m', split="train")
dataset_eval = dataset.select(range(5000, 5100))

translator = GoogleTranslator(source='ko', target='zh-CN')

total_meteor = []
google_meteor = []
for i in range(len(dataset_eval)):
    ko = dataset_eval['ko'][i]
    zh = dataset_eval['zh'][i]

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "translate korean to chinese:",  # instruction
                ko,# input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    de_output = tokenizer.batch_decode(outputs)
    res_output = de_output[0][de_output[0].index("Response:") + len("Response:\n"):de_output[0].index("<|end_of_text|>") ]
    google_output = translator.translate(ko)

    reference = jieba.lcut(zh)
    candidate = jieba.lcut(res_output)
    google_candidate = jieba.lcut(google_output)

    meteor = meteor_score([reference], candidate)
    total_meteor.append(meteor)

    g_meteor = meteor_score([reference], google_candidate)
    google_meteor.append(g_meteor)

    print(i)
    print(f"韓文原句子: {ko}")
    print(f"model 翻譯的句子: {res_output}")
    print(f"google 翻譯的句子: {google_output}")
    print(f"正確句子翻譯的:    {zh}")
    print(f"model_scores: {meteor}")
    print(f"google_scores: {g_meteor}")
    print()

final_meteor = sum(total_meteor) / len(total_meteor)
print(f"最終 model meteor_scores: {final_meteor}")
final_google_meteor = sum(google_meteor) / len(google_meteor)
print(f"最終 google meteor_scores: {final_google_meteor}")
#----------------------------------------------------------------------------------------------
# ja Meteor
dataset = load_dataset(path='larryvrh/WikiMatrix-v1-Ja_Zh-filtered', split="train")
dataset_eval = dataset.select(range(5000, 5100))

translator = GoogleTranslator(source='ja', target='zh-CN')

total_meteor = []
google_meteor = []
for i in range(len(dataset_eval)):
    ja = dataset_eval['ja'][i]
    zh = dataset_eval['zh'][i]

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "translate korean to chinese:",  # instruction
                ja,# input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    de_output = tokenizer.batch_decode(outputs)
    res_output = de_output[0][de_output[0].index("Response:") + len("Response:\n"):de_output[0].index("<|end_of_text|>") ]
    google_output = translator.translate(ja)

    reference = jieba.lcut(zh)
    candidate = jieba.lcut(res_output)
    google_candidate = jieba.lcut(google_output)

    meteor = meteor_score([reference], candidate)
    total_meteor.append(meteor)

    g_meteor = meteor_score([reference], google_candidate)
    google_meteor.append(g_meteor)

    print(i)
    print(f"日文原句子: {ja}")
    print(f"model 翻譯的句子: {res_output}")
    print(f"google 翻譯的句子: {google_output}")
    print(f"正確句子翻譯的:    {zh}")
    print(f"model_scores: {meteor}")
    print(f"google_scores: {g_meteor}")
    print()

final_meteor = sum(total_meteor) / len(total_meteor)
print(f"最終 model meteor_scores: {final_meteor}")
final_google_meteor = sum(google_meteor) / len(google_meteor)
print(f"最終 google meteor_scores: {final_google_meteor}")
