## BERT as Language Model

For a sentence <img src="https://www.zhihu.com/equation?tex=S%20=%20w_1,%20w_2,...,%20w_k" alt="S = w_1, w_2,..., w_k" eeimg="1"> , we have

<img src="https://www.zhihu.com/equation?tex=p(S)%20=%20\prod_{i=1}^{k}%20p(w_i%20|%20context)" alt="p(S) = \prod_{i=1}^{k} p(w_i | context)" eeimg="1"> 


In traditional language model, such as RNN,  <img src="https://www.zhihu.com/equation?tex=context%20=%20w_1,%20...,%20w_{i-1}" alt="context = w_1, ..., w_{i-1}" eeimg="1"> , 

<img src="https://www.zhihu.com/equation?tex=p(S)%20=%20\prod_{i=1}^{k}%20p(w_i%20|%20w_1,%20...,%20w_{i-1})" alt="p(S) = \prod_{i=1}^{k} p(w_i | w_1, ..., w_{i-1})" eeimg="1">


In bidirectional language model, it has larger context, <img src="https://www.zhihu.com/equation?tex=context+%3d+w_1%2c+...%2c+w_%7bi-1%7d%2cw_%7bi%2b1%7d%2c...%2cw_k" alt="context = w_1, ..., w_{i-1},w_{i+1},...,w_k" eeimg="1">.

In this implementation, we simply adopt the following approximation,

<img src="https://www.zhihu.com/equation?tex=p(S)+%5capprox+%5cprod_%7bi%3d1%7d%5e%7bk%7d+p(w_i+%7c+w_1%2c+...%2c+w_%7bi-1%7d%2cw_%7bi%2b1%7d%2c+...%2cw_k)" alt="p(S) \approx \prod_{i=1}^{k} p(w_i | w_1, ..., w_{i-1},w_{i+1}, ...,w_k)" eeimg="1">.


<!--
1. 近似相等
2. 句子越长，单个word预测的概率越大，ppl越大？传统的RNN也有这个问题
-->

<!-- n-gram
n-gram models construct tables of conditional probabilities for the next word,

Under Markov assumption, the context is the all the 
-->


### test-case

> [more cases: 中文](cases/test.zh.md)


```bash
export BERT_BASE_DIR=model/uncased_L-12_H-768_A-12
export INPUT_FILE=data/lm/test.en.tsv
python run_lm_predict.py \
  --input_file=$INPUT_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/tmp/lm_output/
```

for the following test case

```bash
$ cat data/lm/test.en.tsv 
there is a book on the desk
there is a plane on the desk
there is a book in the desk

$ cat /tmp/lm/output/test_result.json
```
output:

```yml
# prob: probability
# ppl:  perplexity
[
  {
    "tokens": [
      {
        "token": "there",
        "prob": 0.9988962411880493
      },
      {
        "token": "is",
        "prob": 0.013578361831605434
      },
      {
        "token": "a",
        "prob": 0.9420605897903442
      },
      {
        "token": "book",
        "prob": 0.07452250272035599
      },
      {
        "token": "on",
        "prob": 0.9607976675033569
      },
      {
        "token": "the",
        "prob": 0.4983428418636322
      },
      {
        "token": "desk",
        "prob": 4.040586190967588e-06
      }
    ],
    "ppl": 17.69329728285426
  },
  {
    "tokens": [
      {
        "token": "there",
        "prob": 0.996775209903717
      },
      {
        "token": "is",
        "prob": 0.03194097802042961
      },
      {
        "token": "a",
        "prob": 0.8877727389335632
      },
      {
        "token": "plane",
        "prob": 3.4907534427475184e-05   # low probability
      },
      {
        "token": "on",
        "prob": 0.1902322769165039
      },
      {
        "token": "the",
        "prob": 0.5981084704399109
      },
      {
        "token": "desk",
        "prob": 3.3164762953674654e-06
      }
    ],
    "ppl": 59.646456254851806
  },
  {
    "tokens": [
      {
        "token": "there",
        "prob": 0.9969795942306519
      },
      {
        "token": "is",
        "prob": 0.03379646688699722
      },
      {
        "token": "a",
        "prob": 0.9095568060874939
      },
      {
        "token": "book",
        "prob": 0.013939591124653816
      },
      {
        "token": "in",
        "prob": 0.000823647016659379  # low probability
      },
      {
        "token": "the",
        "prob": 0.5844194293022156
      },
      {
        "token": "desk",
        "prob": 3.3361218356731115e-06
      }
    ],
    "ppl": 54.65941516205144
  }
]
```



