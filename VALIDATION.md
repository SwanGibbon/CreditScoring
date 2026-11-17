# Проверки и воспроизводимость

Зафиксированный эксперимент: `home-credit-full-001`, 2026-11-17.
[Агрегированные результаты и provenance](reports/home-credit-full-001.json)
публикуются вместе с кодом. Dataset, модели, OOF/holdout строки и планы
с индексами остаются локально.

## Проверки на реальных данных

- Проверены размеры и SHA-256 всех восьми CSV, заголовки, row counts,
  бинарность/полнота TARGET и одинаковая predictor schema train/test.
- Application IDs уникальны и не пересекаются, bureau key уникален;
  SK_ID_PREV имеет одного владельца между history-файлами и chunks.
- 3 120 184 неприсоединяемые bureau_balance rows исключены и подсчитаны.
- Ledger обработал 13 605 401 payment rows с partial/missed payments;
  временная SQLite-база после обработки удалена.
- Полный запуск завершился с exit code 0 за 1176 с. Параметры: 3 outer folds,
  600 max rounds, patience 50, probe 100, top-K 500, leaf complexities 15/31,
  4 потока, seed 42. Локальный launcher контролировал свободный диск.
- 246 009 development и 61 502 final holdout-строки не пересекаются.
  Проверены все реальные fit/tune/calibration/outer/final partitions.
- Метки сохранённых прогнозов сверены с исходными строками. Holdout AUC,
  AP, cost, approval, Brier и log loss пересчитаны из замороженных CSV и
  совпали с report с абсолютным допуском 1e-12.
- Сохранённая модель загружается; SHA-256 и metadata совпали. Submission
  содержит 48 744 test IDs в исходном порядке и конечные вероятности в [0,1].
  Submission не отправлялся.
- Bootstrap вход и код проверены по hashes: paired stratified percentile,
  1000 повторов, seed 20261117. Модель и threshold при анализе не менялись.

## Автоматические проверки

Локально пройдено **58 тестов** с установленными optional XGBoost/CatBoost:

```bash
python -m pytest -q
```

Без optional-зависимостей два их теста пропускаются. Набор проверяет:

- Изменение outer/final holdout labels на фиксированных partitions не меняет
  обучение, predictions или decisions.
- Изменение calibration labels не меняет features, iterations или probabilities.
- Vocabulary, фильтры и supervised selector локальны к training partition.
- Group/time splits исключают пересечения и обучение на будущем.
- Порог совпадает с полным перебором, включая ties и крайние решения.
- Partial/missed/future payments, unknown STATUS и границы chunks обрабатываются
  согласованно; конфликт владельцев или schema mismatch останавливают загрузку.
- Сквозной CSV→CLI→submission, сохранение/загрузка модели и notebook export работают.
- Weighted bootstrap AUC с ties совпадает с явным resampling.

Notebook проверен через nbclient: девять code cells выполняются в чистой копии
без датасета. Синтетический путь отделён от чтения реального публичного отчёта.
Перед публикацией убраны служебные synthetic outputs; отчётные таблицы сохранены.
Compileall и git diff --check пройдены.

## Что именно опубликовано

Публичный JSON содержит конфигурацию, версии, числа наблюдений, метрики folds
и holdout, интервалы, агрегированные сравнения и hashes. В нём нет targets,
индивидуальных probabilities, split indices, приватных ссылок или local paths.
Все восемь input hashes совпали с проверенными локальными копиями.

Training-source hashes относятся к исходникам зафиксированного запуска.
Сохранённая локальная source-копия проверена по ним; fitting code совпадает
с публикуемым. Единственное изменение прежнего CLI source после прогона —
уточнение help о development-only submission; оно не затрагивает расчёт.
Скрипты экспорта и отображение notebook добавлены после замораживания результатов.

## Пределы проверки

Это IID benchmark, не forward-time оценка или production lending system.
Absolute application timestamps и надёжные person links отсутствуют.
Bootstrap условен на фиксированных моделях и числах классов; drift и retraining
uncertainty не покрыты. Просмотренный holdout не является новым независимым
подтверждением будущих правок. Cost 10:1 — учебный сценарий.
Полное реальное сравнение XGBoost/CatBoost и оценка скрытых Kaggle test labels
не проводились. Детали — [отчёт](RESULTS.md) и [методика](docs/methodology.md).
