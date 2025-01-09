# 在任何需要获取schema的地方
from src.core.schema.manager import SchemaManager

# 获取schema manager实例
schema_manager = SchemaManager()

try:
    # 获取格式化的schema字符串
    formatted_schema = schema_manager.get_formatted_enriched_schema(
        db_id="debit_card_specializing",
        source="bird_dev"
    )
    print(formatted_schema)
    
except ValueError as e:
    print(f"Error: {str(e)}")


