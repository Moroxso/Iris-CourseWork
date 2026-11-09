# Установка Iris-CourseWork на Windows x64

Инструкция для развёртывания на другом компьютере. Режим показа — локальный,
на центральном процессоре, без CUDA, Docker и WSL. Интернет нужен для клонирования,
установки зависимостей и загрузки модели. После подготовки запросы обрабатываются
локально. Для автоматизированного развёртывания передайте Codex
[контекст проекта](docs/deployment/CODEX_CONTEXT.md).

## 1. Предварительная проверка

- Windows x64; PowerShell и обычный браузер. Права администратора для самого
  проекта и демонстрации не нужны.
- Git и uv. Проверялись CPython 3.12.11 и uv 0.12.0. Пакеты устанавливаются
  из `uv.lock`, без изменения версий и без глобального `pip install`.
- Оставьте не менее **6 ГиБ доступной оперативной памяти** перед запуском.
  На диске заложите **10 ГиБ свободного места** как запас для установки:
  это не измеренный размер комплекта. Файл весов занимает около 1,5 ГБ,
  отдельно место занимают Python, библиотеки и их кэши. Не храните копии весов.

Проверьте наличие инструментов командами `git --version` и `uv --version`.
Если их нет, используйте [Git for Windows](https://gitforwindows.org/) и
[официальную установку uv](https://docs.astral.sh/uv/getting-started/installation/).
При доступном WinGet можно установить отсутствующий инструмент отдельно:

```powershell
winget install --exact --id Git.Git --source winget
winget install --exact --id astral-sh.uv --source winget
```

После установки откройте новую консоль и проверьте версии. Если установщик
требует повышения прав или условия лицензии, решение принимает оператор.
Проверку подписей, антивирус и политику запуска PowerShell не отключайте.

## 2. Получение проекта и зависимостей

Из каталога, в котором хотите разместить проект:

```powershell
git clone -c core.longpaths=true https://github.com/Moroxso/Iris-CourseWork.git
Set-Location Iris-CourseWork
git rev-parse HEAD
uv python install 3.12.11
uv sync --locked --python 3.12.11
```

Если проект уже клонирован, используйте существующую копию без перезаписи
пользовательских файлов. На любом шаге с ненулевым кодом выхода остановитесь
и разберите первую ошибку. `uv sync --locked` должен оставить `uv.lock` неизменным.
Проверки из корня проекта, не загружающие веса:

```powershell
$python = Join-Path (Get-Location).Path '.venv/Scripts/python.exe'
& $python -I -B -c "import platform, torch, transformers; assert platform.python_version_tuple()[:2] == ('3','12'); assert (torch.tensor([1.0], device='cpu') + 1).item() == 2; print(platform.python_version(), torch.__version__, transformers.__version__)"
& ./.venv/Scripts/iris.exe --help
& $python -B -m pytest -q -p no:cacheprovider tests/unit/test_coursework_console.py tests/unit/test_coursework_live_viewer.py tests/unit/test_coursework_offline_kit.py tests/unit/test_coursework_trained_demo.py
```

## 3. Модель и локальный учебный комплект

Снимок [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
привязан к ревизии `c1899de289a04d12100db370d81485cdf75e47ca`.
Команда загружает только семь файлов, нужных этому проекту. Если снимок
уже есть, присвойте его абсолютный путь переменной `$snapshot` и пропустите
загрузку. Последний компонент пути должен быть идентификатором ревизии.
Не копируйте модель в Git и не используйте другую ревизию как замену.

```powershell
$modelPath = & $python -I -B -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']))"
if ($LASTEXITCODE -ne 0) { throw 'Загрузка модели не завершена' }
$snapshot = $modelPath.Trim()
```

Соберите комплект **на целевом ПК**. Он связывает исходники, модель,
интерпретатор, версии библиотек и снимок состояния моста. Копировать комплект
или `.venv` с другого компьютера не нужно. Обучается только мост: шесть
учебных шагов, без изменения весов Qwen. Следующие команды создают новые
каталоги и останавливаются при ошибке:

```powershell
$tag = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$train = Join-Path (Get-Location).Path "artifacts/bridge-$tag"
$kit = Join-Path (Get-Location).Path "artifacts/defence-kit-$tag"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

& $python -I -B scripts/coursework_bridge_train.py --snapshot $snapshot --out $train --steps 6
if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать снимок состояния моста' }
$checkpoint = Join-Path $train 'checkpoint-step-006'
& $python -I -B scripts/coursework_offline_kit.py build --snapshot $snapshot --checkpoint $checkpoint --out $kit
if ($LASTEXITCODE -ne 0) { throw 'Не удалось собрать комплект' }
$manifest = Join-Path $kit 'manifest.json'
& $python -I -B scripts/coursework_offline_kit.py verify --manifest $manifest --snapshot $snapshot --checkpoint $checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Не пройдена проверка комплекта' }
```

Запишите три пути: `$snapshot`, `$checkpoint` и `$manifest`. Они нужны
для повторного запуска. После изменения исходников или среды старый комплект
может не пройти проверку: сначала выясните причину. Собирайте новый комплект
в новом каталоге; если снимок моста больше не совместим, создайте новый мост
по тем же шести шагам.

## 4. Пробный запуск и демонстрация

Из корня проекта, используя сохранённые пути:

```powershell
& $python -I -B scripts/coursework_console.py --snapshot $snapshot --checkpoint $checkpoint --manifest $manifest --live
```

Этот запуск не требует изменения политики Windows для файлов `.ps1`.
После «Готово» откройте локальный адрес из консоли и введите:

> В записке указан пароль «кедр». Какой пароль указан? Ответь одним словом.

Проверьте, что консоль выдала ответ, на схеме доступны три цикла,
в каждом есть восемь модулей, восемь слотов и две записи. В первом цикле
читать ещё нечего. Выберите третий цикл и модуль чтения, затем раскройте
одну победившую запись: должны быть доступны вектор заявки и вектор
после записи, по 256 чисел каждый. Это не слова. Отдельная таблица
показывает восемь векторов, переданных на вход генератора.

Выйдите командой `/exit` и откройте `viewer.html` сохранённого запроса
в каталоге сеанса. Он работает без запущенной консоли. Адрес живой
вкладки доступен только на этом ПК, пока идёт сеанс; не публикуйте его.

Если ответ модели неверен, сохраните фактический результат. Это не повод
подменять ответ, менять веса или объявлять установку неработоспособной:
проверка запуска и оценка качества ответа — разные задачи.
