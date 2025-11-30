
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-7B",local_dir = "LLM_model/Qwen-LoRA",ignore_patterns=["*.h5","flax_model.msgpack","original/*"], local_dir_use_symlinks=False,resume_download=True)
print("预训练模型已下载到Qwen-NER-LoRA/LLM_model目录下对应子文件夹中")
