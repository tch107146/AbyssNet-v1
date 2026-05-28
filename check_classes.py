import json

# 請換成你實際的 JSON 檔案路徑
json_path = r'D:\DEIMv2-main\train\_annotations.coco.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

categories = data.get('categories', [])
print(f"總共有 {len(categories)} 個類別")
for cat in categories:
    print(f"ID: {cat['id']}, Name: {cat['name']}")