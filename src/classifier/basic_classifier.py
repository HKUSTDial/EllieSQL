import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# 读取 JSONL 文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 提取特征
def extract_features(data):
    features = []
    labels = []
    for item in data:
        schema_info = item["enhanced_linked_schema_wo_info"]
        tables = schema_info["tables"]
        num_tables = len(tables)
        num_columns = sum(len(table["columns"]) for table in tables)
        
        features.append([num_tables, num_columns])
        label = item["pipeline_type"]
        if label in ["ADVANCED", "UNSOLVED"]:
            label = "ADV_UNSOLVED"
        labels.append(label)
    
    return features, labels

# 加载训练集和测试集数据
train_file_path = "./data/labeled/nonempty_bird_train_pipeline_label.jsonl"
test_file_path = "./data/labeled/bird_dev_pipeline_label.jsonl"
train_data = load_jsonl(train_file_path)
test_data = load_jsonl(test_file_path)

# 提取特征和标签
X_train, y_train = extract_features(train_data)
X_test, y_test = extract_features(test_data)

# 1 训练随机森林分类器
print("正在训练随机森林分类器。。。")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测并评估模型
print("预测并评估模型")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 统计预测结果分布
print("预测结果分布:", Counter(y_pred))
print("########################################")



# 2. 训练决策树分类器
print("正在训练决策树分类器。。。")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 预测并评估模型
print("预测并评估模型")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 统计预测结果分布
print("预测结果分布:", Counter(y_pred))
print("########################################")

# # 3. 训练SVM分类器
# print("正在训练SVM分类器。。。")
# clf = SVC(kernel='linear', random_state=42)
# clf.fit(X_train, y_train)

# # 预测并评估模型
# print("预测并评估模型")
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率: {accuracy:.4f}")

# # 统计预测结果分布
# print("预测结果分布:", Counter(y_pred))
print("########################################")


# 4. 训练KNN分类器
print("正在训练KNN分类器。。。")
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train, y_train)

# 预测并评估模型
print("预测并评估模型")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 统计预测结果分布
print("预测结果分布:", Counter(y_pred))
print("########################################")


# 5. 训练朴素贝叶斯分类器
print("正在训练朴素贝叶斯分类器。。。")
clf = GaussianNB()
clf.fit(X_train, y_train)

# 预测并评估模型
print("预测并评估模型")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 统计预测结果分布
print("预测结果分布:", Counter(y_pred))
print("########################################")

# # 6. 训练极限梯度提升（XGBoost）分类器
# print("正在训练极限梯度提升（XGBoost）分类器。。。")
# clf = XGBClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # 预测并评估模型
# print("预测并评估模型")
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率: {accuracy:.4f}")

# # 统计预测结果分布
# print("预测结果分布:", Counter(y_pred))
# print("########################################")



# # 7. 训练逻辑回归（Logistic Regression）分类器
# print("正在训练逻辑回归（Logistic Regression）分类器。。。")
# clf = LogisticRegression(max_iter=200, random_state=42)
# clf.fit(X_train, y_train)

# # 预测并评估模型
# print("预测并评估模型")
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率: {accuracy:.4f}")

# # 统计预测结果分布
# print("预测结果分布:", Counter(y_pred))
# print("########################################")








# # 预测新样本的函数
# def predict_pipeline_type(enhanced_linked_schema_wo_info):
#     num_tables = len(enhanced_linked_schema_wo_info["tables"])
#     num_columns = sum(len(table["columns"]) for table in enhanced_linked_schema_wo_info["tables"])
    
#     feature_vector = [[num_tables, num_columns]]
#     return clf.predict(feature_vector)[0]

# # 示例预测
# sample_schema = {"tables": [{"table": "molecule", "columns": ["molecule_id", "label"]}, {"table": "atom", "columns": ["atom_id", "molecule_id", "element"]}]}
# print("预测的 pipeline_type:", predict_pipeline_type(sample_schema))
