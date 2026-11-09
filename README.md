# Iris-CourseWork

Учебный прототип языковой модели с глобальным рабочим пространством.
Предобученная Qwen3-0.6B заморожена. После первого прохода восемь
рекуррентных модулей предлагают записи в общее пространство из восьми
слотов. За цикл записываются до двух слотов. После трёх циклов
`LatentBridge` передаёт восемь векторов на вход генератора.

В консоли можно отправить новый запрос и в локальной вкладке увидеть,
какие модули записали данные в слоты. История чата не накапливается.
Это прототип для демонстрации архитектуры: небольшое учебное обучение
моста не гарантирует правильность ответов.

## Установка

Для развёртывания на другом компьютере используйте [полную инструкцию](INSTALL.md).
Если установку выполняет Codex, передайте ему
[контекст развёртывания](docs/deployment/CODEX_CONTEXT.md) и
[список проверок](docs/deployment/ACCEPTANCE.md).

Понадобятся Git, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/)
и место для зависимостей. Полный сценарий ниже рассчитан на Windows x64
с CPU; при запуске с моделью нужно не менее 6 ГиБ доступной оперативной
памяти. Поддержка CUDA этим сценарием не проверена. Сеть нужна для
первоначальной установки и получения модели, а не для последующих запросов.

```powershell
git clone -c core.longpaths=true https://github.com/Moroxso/Iris-CourseWork.git
Set-Location Iris-CourseWork
uv sync --locked --python 3.12
uv run --locked iris --help
```

Для проверки кода без загрузки весов:

```powershell
uv run --locked pytest -q tests/unit/test_coursework_console.py tests/unit/test_coursework_live_viewer.py tests/unit/test_coursework_offline_kit.py
```

## Интерактивная демонстрация

**1. Подготовьте веса.** Используется
[Qwen3-0.6B фиксированной ревизии](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
Команда ниже скачивает нужные файлы в локальный кэш Hugging Face. Если
снимок уже есть, можно сразу присвоить `$snapshot` его абсолютный путь;
имя последней папки должно совпадать с ревизией.

```powershell
$snapshot = (uv run --locked python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']))").Trim()
```

**2. Один раз обучите учебный мост и соберите локальный комплект.**
Модель и остальные модули не обновляются. Чекпойнт и комплект остаются
на вашем ПК: комплект проверяет хеши файлов, конфигурацию среды и
доступность локальной модели. Если каталоги уже существуют, используйте
другие имена; существующие результаты не перезаписываются.

```powershell
$train = Join-Path $PWD 'artifacts/bridge-demo'
$kit = Join-Path $PWD 'artifacts/defence-kit'
$python = Join-Path $PWD '.venv/Scripts/python.exe'

& $python -I -B scripts/coursework_bridge_train.py --snapshot $snapshot --out $train --steps 6
$checkpoint = Join-Path $train 'checkpoint-step-006'
& $python -I -B scripts/coursework_offline_kit.py build --snapshot $snapshot --checkpoint $checkpoint --out $kit
$manifest = Join-Path $kit 'manifest.json'
```

**3. Запустите консоль.** Для этого не нужно разрешать выполнение `.ps1`
в Windows:

```powershell
& $python -I -B scripts/coursework_console.py --snapshot $snapshot --checkpoint $checkpoint --manifest $manifest --live
```

После «Готово» откройте адрес `127.0.0.1` из консоли и введите запрос,
например: «В записке указан пароль «кедр». Какой пароль указан?
Ответь одним словом.» Выберите цикл: сплошные стрелки показывают записи
победивших модулей, пунктир — чтение слотов выбранным модулем перед записью.
Раскройте победившую заявку, чтобы сравнить предложенный и сохранённый
векторы по 256 координат. Это не слова запроса. Восемь финальных векторов
появляются перед генерацией ответа. `/help` показывает ограничения,
`/exit` завершает сеанс. Для повторного просмотра откройте
`viewer.html` в каталоге соответствующего запроса.

Если файлы, путь или версии окружения изменились, проверка не запустит
модель с устаревшим комплектом. Сначала восстановите согласованные
файлы или соберите новый комплект из подходящего чекпойнта.

## Состав репозитория

- `src/iris/` — модель, рабочее пространство, генерация, CLI и конфигурации.
- `scripts/` — обучение моста, проверка комплекта, консоль и просмотр трассы.
- `docs/examples/`, `docs/protocols/` — учебные примеры и машинные спецификации.
- `docs/deployment/` — контекст Codex, профиль среды и проверка готовности.
- `tests/` — тесты и небольшие проверочные данные; отдельные интеграционные
  проверки требуют указанной в них среды или модели.

Кэши, виртуальное окружение, веса, готовые чекпойнты, результаты
прогонов и внутренние ADR не публикуются.

Код Iris лицензирован по [MIT](LICENSE). Предобученные веса скачиваются
отдельно и распространяются на условиях
[лицензии Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE).
