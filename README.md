# AIXueTeacher

基于 TRL 的 DPO 算法，后训练 AIXueTeacher 解决教学决策中的 bad case 问题。

## 快速开始

### 1. 数据处理
```bash
python src/dpo/format_data.py \
    --input datasets/aixue_bad_case/all_bad_cases.json \
    --output datasets/aixue_dpo_dataset \
    --format dataset
```

### 2. DPO 训练
```bash
bash scripts/train_dpo.sh
```

## 数据集

- `datasets/aixue_bad_case/`: 原始 bad case 数据
- `datasets/aixue_dpo_dataset/`: 处理后的 DPO 训练数据

## 附录

### DPO 公式推导

![dpo_formula_derivation_1](README.assets/dpo_formula_derivation_1.jpg) 
![dpo_formula_derivation_2](README.assets/dpo_formula_derivation_2.jpg) 
![dpo_formula_derivation_3](README.assets/dpo_formula_derivation_3.jpg) 
