# AIXueTeacher
基于 TRL 的 DPO 算法，后训练 AIXueTeacher，解决 bad case 中出现的问题。

## 数据集

datasets/repeat_cases.json: 任务多次重复 bad case

datasets/bad_cases.json：幻觉、超出4次等 bad case

## 附录

### DPO 公式推导

![dpo_formula_derivation_1](README.assets/dpo_formula_derivation_1.jpg) 

![dpo_formula_derivation_2](README.assets/dpo_formula_derivation_2.jpg) 

![dpo_formula_derivation_3](README.assets/dpo_formula_derivation_3.jpg) 
