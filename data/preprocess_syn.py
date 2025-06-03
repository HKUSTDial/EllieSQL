import json

"""
spider_syn is a derivative dataset of spider, with a one-to-one correspondence to the objects in spider_dev. 
Its questions are synonymous with those in spider_dev.
This script converts spider_syn into a format suitable for running in this project.
"""

# load spider_syn file
with open('data/spider/spider_syn.json', 'r', encoding='utf-8') as f:
    syn_data = json.load(f)

# Create an index to facilitate searching.
syn_index = {(item["db_id"], item["SpiderQuestion"]): item["SpiderSynQuestion"] for item in syn_data}

# load spider_dev file
with open('data/formatted_spider_dev.json', 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)

# build the formatted_spider_syn.json file
output_data = []
for item in formatted_data:
    db_id = item["db_id"]
    spider_question = item["question"]

    key = (db_id, spider_question)
    if key in syn_index:
        syn_question = syn_index[key]
    else:
        syn_question = ""  # If no corresponding item is found, the key is still retained but left empty.

    new_item = item.copy()
    new_item["SpiderQuestion"] = new_item.pop("question")
    new_item["SpiderSynQuestion"] = syn_question
    output_data.append(new_item)

# write into formatted_spider_syn.json
with open('data/formatted_spider_syn.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Conversion completed, output has been written into formatted_spider_syn.json")
