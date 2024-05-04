import os
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
if __name__ == "__main__":
    # pip install --no-binary=protobuf 'protobuf<=3.20.1' --force-reinstall
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    llama_tokenizer_dir = r'yaofu/llama-2-7b-80k'  # 这里是LLaMA tokenizer的路径
    chinese_tokenizer_dir = r'hpcai-tech/Colossal-LLaMA-2-7b-base'  # 这里是Chinese tokenizer的路径

    # 加载tokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)  # 加载LLaMA tokenizer
    chinese_tokenizer = LlamaTokenizer.from_pretrained(chinese_tokenizer_dir)  # 加载Chinese LLaMA tokenizer

    llama_spm = sp_pb2_model.ModelProto()  # 定义LLaMA tokenizer的sentencepiece model
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())  # 从LLaMA tokenizer中加载sentencepiece model
    chinese_spm = sp_pb2_model.ModelProto()  # 定义Chinese tokenizer的sentencepiece model
    chinese_spm.ParseFromString(chinese_tokenizer.sp_model.serialized_model_proto())  # 从Chinese tokenizer中加载sentencepiece model

    # 输出tokens的信息
    print(len(llama_tokenizer), len(chinese_tokenizer))  # 两个tokenizer的词表大小；
    print(llama_tokenizer.all_special_tokens)  # LLaMA tokenizer的special tokens；输出为['']
    print(llama_tokenizer.all_special_ids)  # LLaMA tokenizer的special tokens对应的id；输出为[0]
    print(llama_tokenizer.special_tokens_map)  # LLaMA tokenizer的special tokens；输出为{'bos_token': '', 'eos_token': '', 'unk_token': ''}

    # 将Chinese tokenizer的词表添加到LLaMA tokenizer中（合并过程）
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)  # LLaMA tokenizer的词表
    print(len(llama_spm_tokens_set))  # LLaMA tokenizer的词表大小；输出为32000

    chinese_spm_tokens_set = set(p.piece for p in chinese_spm.pieces)  
    print(len(chinese_spm_tokens_set))

    difference = llama_spm_tokens_set - chinese_spm_tokens_set
    print('Differnce: ', difference)  # empty set
