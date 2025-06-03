import json

# 读取文件1
with open('data/spider/spider_syn.json', 'r', encoding='utf-8') as f:
    syn_data = json.load(f)

# 创建一个索引，方便查找
syn_index = {(item["db_id"], item["SpiderQuestion"]): item["SpiderSynQuestion"] for item in syn_data}

# 读取文件2
with open('data/formatted_spider_dev.json', 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)

# 构建输出数据
output_data = []
for item in formatted_data:
    db_id = item["db_id"]
    spider_question = item["question"]

    key = (db_id, spider_question)
    if key in syn_index:
        syn_question = syn_index[key]
    else:
        syn_question = ""  # 若找不到对应，同样保留键但留空

    new_item = item.copy()
    new_item["SpiderQuestion"] = new_item.pop("question")
    new_item["SpiderSynQuestion"] = syn_question
    output_data.append(new_item)

# 写入输出文件3
with open('data/formatted_spider_syn.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("转换完成，输出已写入 formatted_spider_syn.json")
