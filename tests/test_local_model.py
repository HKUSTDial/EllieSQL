import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm import LLMBase

async def test_local_model():
    """测试本地模型推理"""
    llm = LLMBase()
    
    # 测试用例
    test_cases = [
        # 简单问答
        {
            "messages": [
                {'role': 'system', 'content': 'You are a SQLite expert.'}, 
                {'role': 'user', 'content': """\nYou are a intelligent and responsible SQLite expert. \n\n### Instruction:\nYou need to read the database schema to generate SQL query for the user question. The outputted SQL must be surrounded by ```sql``` code block.\n\n### Database Schema:\nTable: users\nColumns:\n  - Id: Type: INTEGER | Description: the user id | Value examples: -1, 2, 3\n  - Age: Type: INTEGER | Description: user's age | Value description: \x95 teenager: 13-18\n\x95 adult: 19-65\n\x95 elder: > 65 | Value examples: 37, 35, 28\nPrimary key: Id\n\nTable: posts\nColumns:\n  - OwnerUserId: Type: INTEGER | Meaning: Owner User Id | Description: the id of the owner user | Value examples: 8, 24, 18\n  - Score: Type: INTEGER | Description: the score of the post | Value examples: 23, 22, 54\n  - Id: Type: INTEGER | Description: the post id | Value examples: 1, 2, 3\n  - ParentId: Type: INTEGER | Meaning: ParentId | Description: the id of the parent post | Value description: commonsense evidence:\nIf the parent id is null, the post is the root post. Otherwise, the post is the child post of other post. | Value examples: 3, 7, 6\n  - LastEditorUserId: Type: INTEGER | Meaning: Last Editor User Id | Description: the id of the last editor | Value examples: 88, 183, 23\nPrimary key: Id\nForeign keys:\n  - ParentId -> posts.Id\n  - OwnerUserId -> users.Id\n  - LastEditorUserId -> users.Id\n\n\n### Hint:\nelder users refers to Age > 65; Score of over 19 refers to Score > = 20\n\n### User Question:\nAmong the posts owned by an elder user, how many of them have a score of over 19?\n\nGenerate the corresponding SQL query surrounded by ```sql``` code block.\n"""}
            ]
        },
        {
            "messages": [
                {'role': 'system', 'content': 'You are a SQLite expert.'}, 
                {'role': 'user', 'content': """\nYou are a intelligent and responsible SQLite expert. \n\n### Instruction:\nYou need to read the database schema to generate SQL query for the user question. The outputted SQL must be surrounded by ```sql``` code block.\n\n### Database Schema:\nTable: users\nColumns:\n  - Id: Type: INTEGER | Description: the user id | Value examples: -1, 2, 3\n  - Age: Type: INTEGER | Description: user's age | Value description: \x95 teenager: 13-18\n\x95 adult: 19-65\n\x95 elder: > 65 | Value examples: 37, 35, 28\nPrimary key: Id\n\nTable: posts\nColumns:\n  - OwnerUserId: Type: INTEGER | Meaning: Owner User Id | Description: the id of the owner user | Value examples: 8, 24, 18\n  - Score: Type: INTEGER | Description: the score of the post | Value examples: 23, 22, 54\n  - Id: Type: INTEGER | Description: the post id | Value examples: 1, 2, 3\n  - ParentId: Type: INTEGER | Meaning: ParentId | Description: the id of the parent post | Value description: commonsense evidence:\nIf the parent id is null, the post is the root post. Otherwise, the post is the child post of other post. | Value examples: 3, 7, 6\n  - LastEditorUserId: Type: INTEGER | Meaning: Last Editor User Id | Description: the id of the last editor | Value examples: 88, 183, 23\nPrimary key: Id\nForeign keys:\n  - ParentId -> posts.Id\n  - OwnerUserId -> users.Id\n  - LastEditorUserId -> users.Id\n\n\n### Hint:\nelder users refers to Age > 65; Score of over 19 refers to Score > = 20\n\n### User Question:\nAmong the posts owned by an elder user, how many of them have a score of over 19?\n\nGenerate the corresponding SQL query surrounded by ```sql``` code block.\n"""}
            ]
        }
    ]
    
    # 测试不同的本地模型
    models = ["qwen2.5-coder-7b-instruct"]  # 可以添加更多模型
    
    for model in models:
        print(f"\n=== Testing {model} ===")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[Test Case {i}]")
            # print("Input:")
            # for msg in test_case["messages"]:
            #     print(f"{msg['role']}: {msg['content']}")
            
            try:
                response = await llm.call_llm(
                    messages=test_case["messages"],
                    model=model,
                    temperature=0.7,  # 增加一些随机性
                    max_tokens=500
                )
                
                print("\nOutput:")
                print(response["response"])
                print(f"\nTokens: input={response['input_tokens']}, "
                      f"output={response['output_tokens']}, "
                      f"total={response['total_tokens']}")
                
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    asyncio.run(test_local_model())

if __name__ == "__main__":
    main() 