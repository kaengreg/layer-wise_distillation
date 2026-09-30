# Отчёт о работе с AI-агентом

## Общая информация

- Агент: Codex
- Репозиторий: `kaengreg/layer-wise_distillation`
- Ветка: `codex/iterative-pruning`
- Базовый commit: `3a9b5b2` (`Add agent instructions and task specification`)
- Итоговые commits: `4f0c3ee` (`Implement reviewed iterative pruning pipeline`), `72f0efa` (`Expand offline CPU regression coverage`)
- Локальная среда проверки артефактов: macOS, Python 3.11, PyTorch 2.3.1, Transformers 4.57.6, CPU
- GPU-среда: один NVIDIA H100, BF16; полный environment manifest в переданных артефактах отсутствует
- Артефакты эксперимента: `gpu_exp/outputs/qwen2.5-3b-two-layer-pruning/`
- Лог обучения: `gpu_exp/logs/qwen2.5-3b-two-layer-pruning.log`

## Итерация 1: первоначальная реализация

- Промпт: `prompts/00-initial-implementation.md`
- Цель: реализовать воспроизводимый цикл Block Influence → pruning → локальный repair distillation до заданной глубины, сохраняющий checkpoint и метаданные каждой итерации.
- Что сделал агент: добавил пакет `layerwise_distillation` с CLI, padding-aware Block Influence, pruning с отображением текущих слоёв на исходные слои teacher, KL/hidden/LM losses, выбор repair-слоёв, сохранение checkpoint, JSON-метаданных и диагностик, reload validation и обёртку LLMTF.
- Какие файлы изменил: `layerwise_distillation/`, `README.md`, `docs/evaluation-protocol.md`, `.gitignore`, `requirements.txt`, `environment.yml`.
- Результаты тестов: после последующего ревью итоговый CPU-набор содержит 29 тестов; тесты не обращаются к сети и не требуют CUDA.
- Обнаруженные проблемы: первоначальная реализация не была зафиксирована отдельным commit до code review, поэтому реализация и исправления code review находятся вместе в `4f0c3ee`.
- Commit: `4f0c3ee`.

## Итерация 2: ревью реализации

- Промпт: `prompts/01-code-review.md`
- Найденные и подтверждённые проблемы:
  - causal LM mask учитывал только target token и допускал переход из left padding в первый настоящий token;
  - hidden-state loss мог использовать неоднозначные элементы стандартного `hidden_states` tuple вместо выходов decoder blocks;
  - multi-process запуск мог одновременно писать в один каталог;
  - повторный запуск мог смешать новые результаты со старыми артефактами;
  - non-finite loss, gradient, score или metric могли быть сохранены как успешный результат;
  - некоторые конфигурации не имели ни одного gradient path;
  - reload validation не проверял tokenizer;
  - LLMTF wrapper предполагал, что финальным всегда является `iteration_002`;
  - diagnostic perplexity могла молча насыщаться при переполнении.
- Исправления: переходная padding mask, decoder hooks для block outputs, проверка mapping, запрет distributed launch, требование пустого output directory, finite checks, проверка gradient configuration, reload модели и tokenizer с forward pass, определение финальной pruning iteration по метаданным, явная фиксация perplexity overflow.
- Добавленные tests: focused tests для padding, mapping, config repair, distributed guards, reload и invalid configurations; полный список находится в `docs/code-review-report.md`.
- Commit: `4f0c3ee`.

## Итерация 3: ревью тестов

- Промпт: `prompts/02-test-review.md`
- Какие пробелы были найдены: отсутствие end-to-end CPU-проверки двух итераций на реальном Qwen2 class, отсутствие доказательства повторного вычисления importance, слабая проверка неизменности teacher, checkpoint reloadability и реального `datasets.Dataset` schema path, возможность незаметного сетевого доступа.
- Какие тесты добавлены: двухитерационный tiny-Qwen pipeline, проверка importance на глубинах 4 и 3, byte-for-byte неизменность teacher, различие pruning-only/distilled весов, reload всех checkpoint и tokenizer, metadata contract, trainable parameter selection, in-memory Hugging Face Dataset и autouse network blocker.
- Итоговый результат тестов: `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q` → `29 passed`; дополнительно успешно выполнены `python -m compileall -q layerwise_distillation tests` и `git diff --check`.
- Commit: `72f0efa`.

## Итерация 4: H100-эксперимент

- Промпт: `prompts/03-gpu-results-analysis.md`
- Команда запуска:

```bash
CUDA_VISIBLE_DEVICES=0 python -m layerwise_distillation.run \
    --teacher_model_path Qwen/Qwen2.5-3B \
    --dataset kaengreg/ru-miracl-cleaned \
    --dataset_split train \
    --text_column text \
    --target_num_layers 34 \
    --layers_per_iteration 1 \
    --max_importance_samples 512 \
    --max_sequence_length 512 \
    --max_train_samples 20000 \
    --max_eval_samples 1000 \
    --max_train_steps 200 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --protect_first_layers 1 \
    --protect_last_layers 1 \
    --repair_radius 1 \
    --temperature 2.0 \
    --kl_weight 1.0 \
    --hidden_weight 1.0 \
    --lm_weight 0.0 \
    --dtype bfloat16 \
    --attention_implementation sdpa \
    --seed 1337 \
    --report_to json \
    --output_dir outputs/qwen2.5-3b-two-layer-pruning
```

- Продолжительность: продолжительность обучения не записана в предоставленном логе, поэтому её нельзя восстановить достоверно. Полный LLMTF benchmark занял приблизительно 19 минут 27 секунд от начала teacher evaluation (`18:25:59`) до завершения distilled evaluation (`18:45:26`), включая переключение моделей.
- Конфигурация: Qwen2.5-3B, 36→34 слоя за две итерации, 200 optimizer steps на итерацию, seed 1337, BF16, SDPA, один H100.
- Commit кода GPU-запуска: checkout предположительно соответствовал `72f0efa`, но SHA кода не был записан внутрь experiment artifacts, поэтому это не подтверждается самими артефактами.
- Commit LLMTF: `d36543888b6cc3865cf3a584b5c1bda0b0455567`.
- Conversation config и few-shot count: `conversation_configs/default_foundational.json`, foundational mode, 5-shot, vLLM, tensor parallel size 1, context 8192, batch size 8, temperature 0, repetition penalty 1, internal seed 555.
- Проверенные checkpoint: teacher `Qwen/Qwen2.5-3B`; pruning-only `iterations/iteration_002/pruning_only`; distilled `final`.
- LLMTF smoke evaluation: выполнен только для teacher (8 samples/task, enMMLU accuracy 0.6667, Daru ROUGE-L 0.2717). Артефактов pruning-only и distilled и `smoke/comparison.json` нет, поэтому smoke comparison считается незавершённым.
- LLMTF полный benchmark: завершён для teacher, pruning-only и distilled на всех семи задачах. Число примеров совпадает между моделями: Daru 1000, copy tasks по 100, FLORES по 987, MMLU по 1000.
- Каталоги исходных LLMTF-артефактов: `gpu_exp/outputs/qwen2.5-3b-two-layer-pruning/llmtf/full/{teacher,pruning_only,distilled}`; protocol и сводная таблица: `llmtf/full/{protocol.json,comparison.json}`.
- Commit отчёта: не создан в рамках анализа.

### Проверка технической корректности

Все утверждения в этом разделе получены непосредственно из переданных файлов.

- Достигнута требуемая глубина: iteration 1 уменьшила модель 36→35, iteration 2 — 35→34; `target_num_layers=34`.
- На первой итерации удалён current layer 21, соответствующий original teacher layer 21. На второй итерации снова удалён current layer 21, после обновления mapping соответствующий original teacher layer 22.
- Block Influence пересчитан: первая итерация содержит 36 scores, вторая — 35. Удалённые слои являются первыми в соответствующих детерминированно отсортированных ranking; их scores равны 0.0266827 и 0.0302737.
- Mapping согласован: после двух удалений `original_layer_ids` содержит 34 уникальных значения и исключает original layers 21 и 22.
- На обеих итерациях repair layers равны `[20, 21]`, выполнено по 200 optimizer steps.
- Каждый iteration record в `training_metrics.json` идентичен своему `iterations/iteration_XXX/metadata.json`; configuration и seed совпадают с `run_config.json`.
- Все Block Influence scores, training losses, validation metrics и LLMTF aggregate metrics конечны. Для каждой итерации `loss = kl_loss + hidden_loss + lm_loss` с учётом численной точности; `lm_loss=0`, как задано конфигурацией.
- Конфигурации pruning-only, iteration-2 distilled и final checkpoint имеют `num_hidden_layers=34`, 34 значения `layer_types`, `max_window_layers=34`, BF16 и два присутствующих safetensor shards. Index каждого checkpoint содержит 410 tensors.
- `final/` и `iteration_002/distilled/` byte-for-byte совпадают для config, generation config, index, обоих safetensor shards и tokenizer files.
- H100 run записал успешный reload модели и tokenizer с forward pass в `reload_validation.json`.
- Независимая локальная проверка загрузила `final/` через `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, dtype=torch.bfloat16)`: получен `Qwen2ForCausalLM` с 34 слоями, attention `layer_idx` 0–33; CPU forward pass дал logits shape `[1, 3, 151936]`, все logits конечны.
- Сохранённый `comparison.json` точно восстанавливается из исходных LLMTF `*_total.jsonl`; task sets всех трёх моделей совпадают.

### Измеренные training diagnostics

Значения ниже являются измерениями pipeline, а не интерпретацией downstream quality. Peak memory имеет разный operational scope: pruning-only значение снято до repair, distilled значение включает peak, накопленный во время repair training, поэтому это не чистое сравнение inference memory.

| Модель | Слои | Параметры | Validation loss | Perplexity | Teacher KL | Throughput, token/s | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher | 36 | 3,085,938,688 | 1.861395 | 6.432703 | -3.78e-08 | 8,589.95 | 13.48 GiB |
| Pruning-only | 34 | 2,931,784,704 | 1.915037 | 6.787193 | 0.127829 | 9,039.47 | 13.22 GiB |
| Distilled | 34 | 2,931,784,704 | 1.908714 | 6.744413 | 0.112513 | 8,944.34 | 17.52 GiB |

Удалено 154,153,984 параметра, то есть 4.995% teacher parameters. Distillation относительно pruning-only уменьшила validation loss на 0.006323, perplexity на 0.042781 и teacher KL на 0.015315; inference throughput снизился на 1.05%. Малое отрицательное teacher KL для teacher против самой себя (`-3.78e-08`) является floating-point roundoff около нуля, а не отрицательной дивергенцией в математическом смысле.

Training metrics по итерациям:

| Итерация | Total loss | KL loss | Hidden loss | LM loss | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.142176 | 0.066182 | 0.075994 | 0.000000 | 200 |
| 2 | 0.264547 | 0.120330 | 0.144217 | 0.000000 | 200 |

### Объём обучающих данных относительно исходного подхода

Статья [Iterative Layer-wise Distillation for Efficient Compression of Large Language Models](https://doi.org/10.1134/S1054661826700276) ([доступный полный текст на arXiv](https://arxiv.org/abs/2511.05085)) указывает, что исходная дистилляция использовала корпус на основе `IlyaGusev/rulm`, включающий материалы Wikipedia, литературу, social media и новости. В статье не приведено явное количество training samples. Поэтому число примеров исходного подхода ниже взято не из текста статьи, а из связанного legacy launch script репозитория:

- `multinode-multigpu-scripts/run_ft.sh` задаёт `DSFRAC=500000`, `EPOCHS=1`, `MAXLEN=2048` и запускает `Qwen2Distillation.py` с `--use_local_data true`;
- в этом режиме `Qwen2Distillation.py` интерпретирует `ds_frac` как максимальное число читаемых строк local training file, а затем обучается одну эпоху;
- таким образом, исходная repository configuration была рассчитана максимум на 500,000 строк, или не более 500,000 training examples после отбрасывания невалидных JSON/non-text записей. Это intended sample count legacy-запуска, но не независимо подтверждённое точное число валидных примеров, реально обработанных в опубликованном experiment.

В текущем H100 experiment `max_train_samples=20000`, однако этот параметр задаёт только размер доступного training pool. Фактический объём определяется ограничением по шагам:

```text
200 optimizer steps × 8 gradient-accumulation microbatches × batch size 1
= 1,600 sample presentations на pruning iteration.
```

Было две pruning/distillation iterations, поэтому суммарно выполнено 3,200 sample presentations. `TextBatcher` на каждой итерации создаёт shuffle с тем же seed 1337; поскольку 1,600 меньше pool из 20,000 текстов, обе итерации, вероятнее всего, использовали один и тот же первый набор из 1,600 shuffled examples. Следовательно, число уникальных training examples во всём experiment, вероятнее всего, близко к 1,600, хотя суммарное число предъявлений равно 3,200.

Сравнение с intended legacy configuration при допущении 500,000 валидных samples:

| Величина | Текущий H100 experiment | Исходная repository configuration | Отношение текущего к исходному |
| --- | ---: | ---: | ---: |
| Настроенный training pool | 20,000 | до 500,000 | 4.0% (в 25 раз меньше) |
| Sample presentations за весь run | 3,200 | до 500,000 за одну эпоху | 0.64% (в 156 раз меньше) |
| Вероятные уникальные samples | около 1,600 | до 500,000 | около 0.32% (в 312 раз меньше) |

Кроме sample count режимы различаются по максимальной длине последовательности (512 против 2048), learning rate (`1e-5` против `1e-4`), источнику данных (`kaengreg/ru-miracl-cleaned` против корпуса на основе `IlyaGusev/rulm`) и loss scaling. Поэтому текущий run является короткой технической проверкой repair distillation, а не вычислительно сопоставимым воспроизведением training regime статьи. Малый effective sample budget является правдоподобным фактором слабого восстановления downstream quality, но по одному run нельзя приписать наблюдаемые регрессии только объёму данных.

### Измеренные LLMTF результаты

В таблице сохранены native metrics каждой задачи: accuracy для MMLU, ROUGE-L для Daru и FLORES в данной LLMTF revision, LCS для copy tasks. Усреднять эти гетерогенные метрики как единый quality score научно некорректно, поэтому `evaluation_results.txt` mean не используется для вывода.

| Задача / metric | Teacher | Pruning-only | Distilled | Distilled − pruning | Distilled − teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| Daru Treeway / ROUGE-L | 0.246768 | 0.236456 | 0.236895 | +0.000439 | -0.009873 |
| Copy document RU / LCS | 0.940000 | 0.410000 | 0.440000 | +0.030000 | -0.500000 |
| Copy paragraph RU / LCS | 0.990000 | 0.860000 | 0.880000 | +0.020000 | -0.110000 |
| FLORES EN→RU / ROUGE-L | 0.531165 | 0.506644 | 0.508639 | +0.001995 | -0.022525 |
| FLORES RU→EN / ROUGE-L | 0.605323 | 0.593395 | 0.594129 | +0.000734 | -0.011193 |
| enMMLU / accuracy | 0.708781 | 0.620921 | 0.618843 | -0.002078 | -0.089938 |
| ruMMLU / accuracy | 0.570025 | 0.501353 | 0.498911 | -0.002442 | -0.071113 |

### Интерпретация

- Pruning двух соседних original layers 21 и 22 дал ожидаемое уменьшение модели примерно на 5% и небольшое увеличение измеренного token throughput, но ухудшил все семь downstream metrics относительно teacher.
- Короткий repair distillation улучшил pruning-only результат на пяти генеративных/copy задачах и одновременно немного ухудшил обе MMLU accuracy. Следовательно, утверждение об общем восстановлении качества не подтверждается: эффект task-dependent.
- Самая крупная регрессия — document copying: LCS снизился с 0.94 у teacher до 0.41 после pruning и восстановился только до 0.44 после distillation. Это нельзя скрывать средним значением по разнородным задачам.
- Диагностические validation loss, perplexity и teacher KL после distillation улучшились относительно pruning-only, что согласуется с целью repair objective, но это улучшение не перенеслось на MMLU.
- Один seed и короткий 200-step repair не позволяют оценить variance или statistical significance. Результаты являются одним controlled run, а не доказательством устойчивого превосходства выбранной pruning/distillation конфигурации.

### Ошибки, неполные проверки и ограничения

- Smoke evaluation завершена только для teacher. Причина остановки не представлена в артефактах; pruning-only/distilled smoke comparison отсутствует. Полный benchmark тем не менее завершён для всех трёх моделей.
- Training log содержит download/load/dataset progress, но не содержит timestamps этапов pruning/training, step-wise loss history или итоговую wall-clock duration. Длительность обучения и динамику loss восстановить нельзя.
- Артефакты не содержат SHA commit кода GPU-run, `pip freeze`, `nvidia-smi` report или checksum manifest. LLMTF commit и фактическая Transformers version 4.57.6 записаны.
- При локальной загрузке Transformers 4.57.6 выдал предупреждение об `incorrect regex pattern` в tokenizer и предложил `fix_mistral_regex=True`. Training и LLMTF использовали один и тот же сохранённый tokenizer, поэтому относительное сравнение моделей остаётся согласованным, но влияние этого предупреждения на абсолютные метрики отдельно не проверялось.
- Iteration-1 checkpoint weights не были переданы; iteration-1 metadata присутствует и согласована, но независимый reload checkpoint первой итерации невозможен.
