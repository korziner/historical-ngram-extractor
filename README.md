# historical-ngram-extractor
Частотные словари сочетаемости за 5 минут на корпусах из 20+ тыс книг

<img width="939" height="609" alt="image" src="https://github.com/user-attachments/assets/42fc4941-55dd-4e75-90b1-206c96f2f5ca" />

```
historical-ngram-extractor -h
Экстрактор уникальных n-грамм с поддержкой дореформенной орфографии

Usage: historical-ngram-extractor [OPTIONS]

Options:
  -n, --ngram-size <NGRAM_SIZE>  [env: N=] [default: 2]
  -f, --min-freq <MIN_FREQ>      [env: MIN_FREQ=] [default: 3]
  -d, --dict-path <DICT_PATH>    [env: DICT_PATH=]
  -l, --language <LANGUAGE>      [env: STEMMER_LANG=] [default: ru]
      --noprogress               
      --threads <THREADS>        Число потоков для rayon (переопределяет RAYON_NUM_THREADS) [env: RAYON_THREADS=]
  -h, --help                     Print help
  -V, --version                  Print version
```
Просмотр частотных словарей таким однострочником:
```
zstdcat 4.MIN_FREQ2_multithread.tsv.zst|awk '$NF>9 && length($0)>33'|column -t|sort -rnk5
zstdcat 5.MIN_FREQ2_multithread.tsv.zst|awk '$NF>9 && length($0)>55'|column -t|sort -rnk6
```
ъ в конце добавлен постобработчиком стеммера не случайно, так повышаеся вероятность ugrep -Z найти строку с ошибками OCR/HTR 
