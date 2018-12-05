
# 总体评价


1. char-level的语言模型，由于词组内的高概率，会使整个句子ppl普遍偏高。
1. 句子间的相对ppl还靠谱。
1. 双向语言模型，单个word/char的概率预测效果较好。

> 建议:
用分词后的中文重新pretrain，然后进行word-level language model predict。


# 中文测试

```bash
export BERT_BASE_DIR=model/chinese_L-12_H-768_A-12
export INPUT_FILE=data/lm/test.zh.tsv
python run_lm_predict.py \
  --input_file=$INPUT_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/tmp/lm_output/
```

以下是部分结果，更多见[result.zh.json](/data/lm/result.zh.json)

```yml
[
  {
    "tokens": [
      {
        "token": "2016",
        "prob": 0.06563900411128998
      },
      {
        "token": "全",
        "prob": 0.4981258511543274
      },
      {
        "token": "国",
        "prob": 0.9088247418403625
      },
      {
        "token": "低",
        "prob": 1.6259804397122934e-05  # 低概率
      },
      {
        "token": "考",
        "prob": 0.4023572504520416
      },
      ...
    ],
    "ppl": 13.400421357093588
  },
 {
    "tokens": [
      {
        "token": "落",
        "prob": 0.1483132392168045
      },
      {
        "token": "霞",
        "prob": 0.42232587933540344
      },
      {
        "token": "与",
        "prob": 0.8615185022354126
      },
      {
        "token": "孤",
        "prob": 0.9975666999816895
      },
      {
        "token": "鹜",
        "prob": 0.5613960027694702
      },
      {
        "token": "齐",
        "prob": 0.18012434244155884
      },
      {
        "token": "跑",
        "prob": 1.3388593288254924e-05   # 低概率
      },
      ...
    ],
    "ppl": 11.983086642867598
  },
```


中文测试样例来源于[百度云dnnlm](https://cloud.baidu.com/product/nlp/dnnlm_cn)

<!--
英文model跑中文 - UNK的影响


export BERT_BASE_DIR=model/uncased_L-12_H-768_A-12
export INPUT_FILE=data/lm/test.zh.tsv
python run_lm_predict.py \
  --input_file=$INPUT_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/tmp/lm_output/

UNK太多，没有多大意义。
-->
