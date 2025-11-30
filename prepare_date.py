
import json

## 改变json文件里数据的格式：把{"sentence":"---"，"entities":["name": "---","type": "GENE","pos": [---,---]]}
## 形式改为{"instruction": "---（固定的一段指令）","input": "Text:---","output": "{\"entity_text\": \"---\"}"}
def json_format_transfer(examples,out_json_path):  # 所有的数据
    instruct = (
    "你是生物医学领域的基因实体识别助手。"
    "给定一句英文句子，请找出其中所有属于基因或相关生物标志物的实体。"
    "请逐行输出 JSON 对象，每行形如 {\"entity_text\": \"...\"}。"
    "如果句子中没有任何此类实体，输出：No entities"
    )
    with open(out_json_path,"w",encoding="utf-8") as f:    
        for example in examples:
            sent=example["sentence"]
            ents=example["entities"]
            if ents:
                found_ents=[]
                for ent in ents:
                    name=ent["name"]
                    found_ents.append(json.dumps({"entity_text":name},ensure_ascii=False))#ensure_ascii=False：不全部转成ASCII字符，保留原始Unicode字符（中文、日文等），更适合人类直接看，也方便大模型看到真实的文本
                output="\n".join(found_ents)
            else:
                output="No entities"
            item={
                "instruction":instruct,
                "input":sent,
                "output":output
            }
            line=json.dumps(item,ensure_ascii=False)
            f.write(line+"\n")

