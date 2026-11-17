# Данные

Источник датасета — [Home Credit Default Risk, Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data).
Получите файлы самостоятельно в рамках своего доступа и применимых правил.
Репозиторий не распространяет данные и не принимает соглашения автоматически.

```text
home-credit-default-risk/
  application_train.csv
  application_test.csv
  bureau.csv
  bureau_balance.csv
  previous_application.csv
  POS_CASH_balance.csv
  installments_payments.csv
  credit_card_balance.csv
```

`HomeCredit_columns_description.csv` полезен как словарь полей,
`sample_submission.csv` — для дополнительной сверки test IDs; они необязательны
для pipeline. Папка с данными исключена из Git.

## Состав зафиксированного набора

| Файл | Байты | Строки |
|---|---:|---:|
| application_train.csv | 166 133 370 | 307 511 |
| application_test.csv | 26 567 651 | 48 744 |
| bureau.csv | 170 016 717 | 1 716 428 |
| bureau_balance.csv | 375 592 889 | 27 299 925 |
| previous_application.csv | 404 973 293 | 1 670 214 |
| POS_CASH_balance.csv | 392 703 158 | 10 001 358 |
| installments_payments.csv | 723 118 349 | 13 605 401 |
| credit_card_balance.csv | 424 582 605 | 3 840 312 |

SHA-256 восьми CSV опубликованы в [агрегированном отчёте](reports/home-credit-full-001.json).
Они идентифицируют использованную копию; оригинальный checksum Kaggle отдельно
не предоставлялся. Сама целостность загрузок, структура и содержимое были проверены.

## Связи и проверки

SK_ID_CURR связывает application с историей. Bureau_balance сначала получает
владельца через bureau.SK_ID_BUREAU. SK_ID_PREV связывает previous, POS, cards
и installments; он проверяется на единственного владельца между файлами/chunks.

В проверенном наборе application train/test имеют 122/121 колонки, одинаковые
predictor schemas, уникальные непересекающиеся IDs; TARGET полный и бинарный.
SK_ID_BUREAU уникален. Порядок test IDs совпал с sample_submission.
Исторические ключи заполнены; положительных/неизвестных опорных дат не найдено.

3 120 184 строки bureau_balance не имеют соответствующего кредита в bureau.
Они исключены и подсчитаны, а не присоединены другому клиенту.
В installments 2 905 строк без наблюдённой даты/суммы платежа; они сохраняются
как неоплаченные взносы по правилам snapshot-таблицы.

Даты истории относительны к текущей заявке. Абсолютной application date и
доказанного person ID нет, поэтому IID-оценка не доказывает временной перенос
или отсутствие скрытых повторных лиц. TARGET означает payment difficulties,
не обязательно юридический дефолт. Подробные правила — в [методике](docs/methodology.md).

```bash
python -m credit_scoring.run --check-data
```

Команда проверяет файлы и заголовки; значения, ключи, владельцы и время
проверяются во время полной загрузки. При отсутствии файлов выводятся все
недостающие пути и возвращается exit code 2.
