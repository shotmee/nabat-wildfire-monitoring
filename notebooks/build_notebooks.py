"""One-off generator for 01_af_binary_fire_detection.ipynb and
02_bs_burn_severity.ipynb. Run once with `python build_notebooks.py` whenever
the skeleton needs regenerating from scratch; day-to-day work happens in the
.ipynb files themselves, not here.

Heavy training cells are gated behind RUN_HEAVY_TRAINING and are meant to run
on Colab/Kaggle with a GPU runtime - never on the local dev machine (no GPU,
disk nearly full). Only cheap smoke tests (tiny synthetic tensors, one
forward+backward pass) execute locally, to catch shape/logic bugs before
burning cloud GPU time.

src/dataio.py, src/rle.py and src/metrics.py are embedded verbatim into each
notebook (see embed_src_cell below) instead of imported via `sys.path` -
on Colab the notebook is opened standalone, with none of the surrounding repo
next to it, so `sys.path.insert(0, '../src')` + `import dataio` has nothing to
find. Embedding keeps both notebooks fully self-contained; re-run this script
after editing any src/*.py to refresh the embedded copies.
"""
from pathlib import Path

import nbformat as nbf

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def embed_src_cell(filename):
    text = (SRC_DIR / filename).read_text(encoding="utf-8")
    header = (
        f"# === встроено из src/{filename} - не редактировать здесь, "
        f"правьте src/{filename} и пересоберите ноутбук через build_notebooks.py ===\n"
    )
    return code(header + text)


def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


KAGGLE_MD = """## 0. Данные: загрузка с Kaggle

Датасет уже опубликован: https://www.kaggle.com/datasets/shotme/cosmohack2026
(публичный, `shotme/cosmohack2026` уже подставлен ниже) — просто выполняйте
ячейки по порядку, ничего менять не нужно.

Ноутбук рассчитан на запуск в Colab/Kaggle Notebook с GPU. Данные тянутся через
`kagglehub.dataset_download` — ничего вручную скачивать/распаковывать не нужно.
Загрузчик ниже сам определяет, лежит ли `fire-{train,test}-renamed` на Kaggle
как исходный `.tar` или как уже распакованная папка (Data Explorer Kaggle
иногда показывает архивы «развёрнутыми» в дереве) — работает в обоих случаях.

**Тест уже выложен** (`fire-test-renamed`, 180 AF + 89 BS чипов, без масок —
их и предсказываем) и добавлен новой версией того же Kaggle-датасета.
`resolve_test_source(DATASET_DIR)` в ячейках инференса подхватывает его точно
так же, как `resolve_train_source` — train. `test/sample_submission.csv` и
`test/meta.csv` лежат в корне тестового архива (не под `af/`/`bs/`, в отличие
от `train/af/meta.csv` и `train/bs/meta.csv`) — читаются через
`test_source.read_csv("sample_submission.csv")`.

**Авторизация нужна обязательно** — Kaggle API требует ключ даже для скачивания
публичных датасетов (проверено: без него `kagglehub.dataset_download` падает
с ошибкой credentials). Самый чистый вариант под Colab — **Colab Secrets**,
делается один раз на весь Google-аккаунт и работает потом в любом ноутбуке
без единой правки кода:

1. В Colab слева — значок 🔑 (Secrets) → **Add new secret**.
2. Добавить `KAGGLE_USERNAME` = ваш ник на Kaggle.
3. Добавить `KAGGLE_KEY` = API-ключ (kaggle.com → Settings → API →
   **Create New Token** — скачается `kaggle.json`, ключ и username там же).
4. Включить переключатель **Notebook access** у обоих секретов.

После этого ячейка ниже отработает сама. `kagglehub.login()` для Colab **не
использовать** — его виджет не блокирует выполнение ячейки, из-за чего
«Run All» уезжает вперёд к скачиванию раньше, чем токен успевает
сохраниться.
"""

KAGGLE_CODE = """KAGGLE_DATASET_HANDLE = "shotme/cosmohack2026"
_LOCAL_DEV_PATH = r"..\\..\\Мониторинг DATA\\Мониторинг DATA"  # только для локальной разработки

import os

if os.path.isdir(_LOCAL_DEV_PATH):
    # локальная машина разработчика - без похода в сеть (данные уже лежат на диске)
    DATASET_DIR = _LOCAL_DEV_PATH
    print("Найдены локальные данные - используем их:", DATASET_DIR)
else:
    # Colab / любая другая машина - тянем публичный датасет с Kaggle.
    # Нужны Kaggle-credentials, даже для публичных датасетов (см. markdown выше) -
    # в Colab это делается один раз через Secrets (значок ключа слева), без правок кода.
    import kagglehub
    try:
        DATASET_DIR = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
    except Exception as e:
        raise RuntimeError(
            "Не удалось скачать датасет с Kaggle - похоже, не настроены credentials. "
            "В Colab: значок ключа (Secrets) слева -> Add new secret -> добавить "
            "KAGGLE_USERNAME и KAGGLE_KEY (взять на kaggle.com -> Settings -> API -> "
            "Create New Token) -> включить Notebook access для обоих -> перезапустить "
            "эту ячейку. Подробности - README.md в solution/."
        ) from e
    print("Датасет с Kaggle загружен в:", DATASET_DIR)


def _resolve_source(dataset_dir, kind, source_classes):
    \"\"\"kind = 'train' или 'test'. fire-{kind}-renamed на Kaggle может быть как исходным
    .tar, так и уже распакованной папкой (зависит от того, разворачивает ли Kaggle архив
    при показе/скачивании) - подхватываем оба варианта без ручных правок.\"\"\"
    TarChipSource, DirChipSource = source_classes
    tar_path = os.path.join(dataset_dir, f"fire-{kind}-renamed.tar")
    if os.path.isfile(tar_path):
        print(f"Источник {kind}: tar-архив ->", tar_path)
        return TarChipSource(tar_path, root=kind)
    for candidate in (
        os.path.join(dataset_dir, f"fire-{kind}-renamed", kind),
        os.path.join(dataset_dir, kind),
    ):
        if os.path.isdir(candidate):
            print(f"Источник {kind}: распакованная папка ->", candidate)
            return DirChipSource(candidate)
    raise FileNotFoundError(
        f"Не нашли ни {tar_path}, ни распакованную папку {kind}/ внутри {dataset_dir} - "
        "проверьте структуру датасета на Kaggle (Data Explorer)."
    )


def resolve_train_source(dataset_dir):
    return _resolve_source(dataset_dir, "train", (TarChipSource, DirChipSource))


def resolve_test_source(dataset_dir):
    \"\"\"Тестовые чипы (fire-test-renamed) - без масок, plus test/sample_submission.csv
    и test/meta.csv лежат в корне (не под af/ или bs/, в отличие от train).\"\"\"
    return _resolve_source(dataset_dir, "test", (TarChipSource, DirChipSource))
"""

HEAVY_GUARD_MD = """## Важно: где обучать

`RUN_HEAVY_TRAINING` ниже управляет тяжёлыми ячейками (обучение LightGBM /
U-Net). На локальной машине держим его `False` — здесь нет GPU и мало места на
диске. Ставим `True` только на Colab (Runtime → Change runtime type → GPU) или
Kaggle Notebook (Settings → Accelerator → GPU T4x2/P100). Всё остальное
(feature engineering, EDA, быстрые smoke-тесты кода модели) выполняется и
локально."""

HEAVY_GUARD_CODE = """import torch

RUN_HEAVY_TRAINING = False  # True только на Colab/Kaggle с GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| RUN_HEAVY_TRAINING:", RUN_HEAVY_TRAINING)
"""

CHECKPOINT_MD = """### Куда сохраняются веса

`/content` в Colab — эфемерная файловая система: всё в ней пропадает при
отключении рантайма (по таймауту или вручную), а обучение занимает часы —
терять веса из-за этого обидно. Поэтому чекпоинты и метаданные для
воспроизводимости (гиперпараметры, имена признаков, seed, метрика на
валидации) пишутся на Google Drive, если ноутбук выполняется в Colab, или в
локальную папку `./checkpoints` в остальных случаях (эта машина, Kaggle
Notebook и т.п.)."""

CHECKPOINT_CODE = """import os


def _setup_checkpoint_dir():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        path = "/content/drive/MyDrive/kosmohack2026/checkpoints"
    except ImportError:
        path = os.path.join(".", "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


CHECKPOINT_DIR = _setup_checkpoint_dir()
print("Чекпоинты сохраняются в:", CHECKPOINT_DIR)
"""


# ---------------------------------------------------------------------------
# Notebook 1: AF module - binary active-fire detection
# ---------------------------------------------------------------------------

af_cells = [
    md(
        "# Модуль 1 — детекция активного горения (AF)\n\n"
        "Вход: AF-чип 256x256 @ 375 м (VIIRS I1-I5 + solar/sensor zenith + valid, "
        "плюс отдельный AUX-слой: landcover, dem, t2m, rh2m, wind_speed).\n\n"
        "Выход: бинарная маска (1 = природное горение, 0 = всё прочее, включая техногенные "
        "термоаномалии).\n\n"
        "Метрика: `F1_af`, микро-усреднение TP/FP/FN по всем чипам, вес 0.35 в общем `Score`.\n\n"
        "Модель: LightGBM на контекстных пиксельных признаках (см. секцию 5) — это тот же "
        "принцип, что и в операционных алгоритмах активного огня (VIIRS/MODIS AF), только "
        "порог заменён на обученный классификатор с богатым контекстом (локальный фон, "
        "метеоусловия, тип подстилающей поверхности), что должно ощутимо поднять F1 относительно "
        "фиксированного порога."
    ),
    md("## 0. Настройка"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.ndimage import uniform_filter\n"
        "from tqdm.auto import tqdm\n\n"
        "RANDOM_SEED = 42\n"
        "np.random.seed(RANDOM_SEED)"
    ),
    md(
        "Ноутбук самодостаточный — код `src/dataio.py`, `src/rle.py`, `src/metrics.py` встроен "
        "прямо сюда (см. следующие 3 ячейки), поэтому запуск в Colab работает при загрузке "
        "одного `.ipynb`-файла, без остального репозитория рядом."
    ),
    embed_src_cell("dataio.py"),
    embed_src_cell("rle.py"),
    embed_src_cell("metrics.py"),
    md(KAGGLE_MD),
    code(KAGGLE_CODE),
    md(HEAVY_GUARD_MD),
    code(HEAVY_GUARD_CODE),
    md(CHECKPOINT_MD),
    code(CHECKPOINT_CODE),
    code(
        "src = resolve_train_source(DATASET_DIR)\n"
        "meta_af = src.meta('af')\n"
        "meta_af.head()"
    ),
    md(
        "## 1. EDA\n\n"
        "- доля горящих пикселей и распределение `n_fire_px` по чипам (дисбаланс — экстремальный);\n"
        "- разбивка по `landcover` из AUX (доля горящих пикселей на пашне/застройке — потенциальный "
        "источник техногенных срабатываний, которые модель должна научиться подавлять сама через фичу);\n"
        "- визуальный осмотр чипа (I4, I4-I5, маска)."
    ),
    code(
        "n_fire = meta_af['n_fire_px'].fillna(0)\n"
        "print('чипов всего:', len(meta_af))\n"
        "print('чипов с горением:', (n_fire > 0).sum(), 'без горения:', (n_fire == 0).sum())\n"
        "print('всего пикселей горения:', int(n_fire.sum()), 'из', len(meta_af) * 256 * 256,\n"
        "      f'({100 * n_fire.sum() / (len(meta_af) * 256 * 256):.4f}% пикселей)')\n"
        "print('медиана по позитивным чипам:', n_fire[n_fire > 0].median())\n"
        "n_fire[n_fire > 0].hist(bins=30)\n"
        "plt.xlabel('n_fire_px (позитивный чип)'); plt.ylabel('число чипов'); plt.title('Распределение размера очага')"
    ),
    code(
        "chip_id = meta_af.loc[meta_af['n_fire_px'] > 20, 'chip_id'].iloc[0]\n"
        "sample = src.load_af(chip_id)\n"
        "I4, I5 = sample['viirs']['I4'], sample['viirs']['I5']\n"
        "mask = sample['mask']\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
        "axes[0].imshow(I4, cmap='inferno'); axes[0].set_title('I4 (яркостная T, K)')\n"
        "axes[1].imshow(I4 - I5, cmap='inferno'); axes[1].set_title('I4 - I5')\n"
        "axes[2].imshow(mask, cmap='gray'); axes[2].set_title(f'маска, {chip_id}')\n"
        "for ax in axes: ax.axis('off')\n"
        "plt.tight_layout()"
    ),
    md(
        "## 2. Baseline: контекстная детекция (яркостная температура vs фон)\n\n"
        "Логика из постановки: пиксель — кандидат, если `I4` и `I4 - I5` одновременно превышают "
        "статистику фона в скользящем окне. Держим как контрольную точку — обученная модель ниже "
        "должна её уверенно обходить."
    ),
    code(
        "def baseline_af_predict(I4, I5, win=21, k_ctx=3.0):\n"
        "    diff = I4 - I5\n"
        "    bg_I4 = uniform_filter(I4, size=win)\n"
        "    bg_diff = uniform_filter(diff, size=win)\n"
        "    std_I4 = np.sqrt(np.clip(uniform_filter(I4**2, size=win) - bg_I4**2, 0, None)) + 1e-3\n"
        "    std_diff = np.sqrt(np.clip(uniform_filter(diff**2, size=win) - bg_diff**2, 0, None)) + 1e-3\n"
        "    cond_ctx = (I4 - bg_I4 > k_ctx * std_I4) & (diff - bg_diff > k_ctx * std_diff)\n"
        "    cond_abs = I4 > 320\n"
        "    return (cond_ctx | cond_abs).astype(np.uint8)\n\n"
        "pred = baseline_af_predict(np.nan_to_num(I4, nan=280.0), np.nan_to_num(I5, nan=280.0))\n"
        "plt.imshow(pred, cmap='gray'); plt.title('baseline prediction'); plt.axis('off')"
    ),
    md(
        "## 3. Валидационный сплит\n\n"
        "`fire_event_id` у AF-чипов пустой (`meta_af['fire_event_id'].nunique() == 0`) — "
        "группировать по нему, как в BS, нельзя. Но `acq_datetime` — всего 78 уникальных "
        "значений на 420 чипов, то есть до 9 чипов делят один и тот же пролёт спутника "
        "(общая калибровка сенсора, общая погодная картина дня). **Случайный сплит по "
        "`chip_id` пропускает чипы одного пролёта и в train, и в val одновременно** — это "
        "утечка, которая завышает валидационную метрику, не отражая обобщение на новые "
        "пожары/пролёты. Группируем по `acq_datetime`, стратифицируя по тому, встречается "
        "ли горение хотя бы в одном чипе пролёта — аналогично тому, как BS группирует по "
        "`fire_event_id`."
    ),
    code(
        "from sklearn.model_selection import train_test_split\n\n"
        "group_has_fire = meta_af.groupby('acq_datetime')['n_fire_px'].apply(lambda s: (s.fillna(0) > 0).any())\n"
        "train_groups, val_groups = train_test_split(\n"
        "    group_has_fire.index, test_size=0.2, random_state=RANDOM_SEED, stratify=group_has_fire.values\n"
        ")\n"
        "train_ids = meta_af.loc[meta_af['acq_datetime'].isin(train_groups), 'chip_id']\n"
        "val_ids = meta_af.loc[meta_af['acq_datetime'].isin(val_groups), 'chip_id']\n"
        "print(len(train_ids), 'train /', len(val_ids), 'val (уникальных пролётов:', len(group_has_fire), ')')"
    ),
    md(
        "## 4. Baseline на валидации (F1_af, micro) — контрольная точка"
    ),
    code(
        "acc = ConfusionAccumulator(classes=(1,))\n"
        "for chip in tqdm(iter_chips(src, 'af', val_ids), total=len(val_ids), desc='baseline eval (val)'):\n"
        "    I4c = np.nan_to_num(chip['viirs']['I4'], nan=280.0)\n"
        "    I5c = np.nan_to_num(chip['viirs']['I5'], nan=280.0)\n"
        "    valid = nodata_to_ignore(chip['mask'], chip['mask_nodata'])\n"
        "    pred_c = baseline_af_predict(I4c, I5c)\n"
        "    acc.update(pred_c, chip['mask'], valid=valid)\n\n"
        "print('baseline F1_af (val, micro):', f1_af(acc))"
    ),
    md(
        "## 5. Признаки для LightGBM (расширенные)\n\n"
        "Физические (I1-I5, разница I4-I5) + **многомасштабный** контекстный фон (z-score в "
        "окнах 9/21/41 — мелкие техногенные горячие точки часто «горячие» локально, но "
        "сливаются с фоном на большом масштабе, и наоборот для крупных пожаров; их отношение "
        "— почти бесплатный дополнительный сигнал) + геометрия наблюдения (solar/sensor "
        "zenith, день/ночь) + **sun-glint прокси** (отношения I1/I2/I3 и средняя дневная "
        "отражательная способность — блики от воды/крыш/панелей днём дают высокую плоскую "
        "reflectance без настоящего горения) + контекст AUX (landcover, dem/t2m/rh2m/wind_speed) "
        "+ **landcover-интеракции** (`z_I4 × пашня`, `z_diff × застройка` — экономят дереву "
        "сплиты на переоткрытие «горячий пиксель на пашне/в городе требует другого порога», "
        "прямая атака на названную в постановке проблему техногенных термоаномалий).\n\n"
        "NaN в I4/I5 (полосы наблюдения) заполняются средним по чипу перед сглаживанием, чтобы "
        "NaN не расползался по соседним валидным пикселям через `uniform_filter`; I1-I3 "
        "(дневные reflectance-каналы, ночью NaN) обнуляются перед расчётом отношений."
    ),
    code(
        "AF_FEATURE_NAMES = [\n"
        "    'I1', 'I2', 'I3', 'I4', 'I5', 'diff_I4_I5',\n"
        "    'z_I4_w9', 'z_diff_w9',\n"
        "    'z_I4_w21', 'z_diff_w21', 'bg_I4', 'bg_diff',\n"
        "    'z_I4_w41', 'z_diff_w41', 'z_I4_scale_ratio',\n"
        "    'solar_zenith', 'sensor_zenith', 'is_night',\n"
        "    'I1_I2_ratio', 'I2_I3_ratio', 'mean_refl_day',\n"
        "    'landcover', 'dem', 't2m', 'rh2m', 'wind_speed',\n"
        "    'z_I4_x_cropland', 'z_diff_x_urban',\n"
        "]\n"
        "AF_LANDCOVER_IDX = AF_FEATURE_NAMES.index('landcover')\n"
        "AF_LANDCOVER_CROPLAND_CODE = 40  # ESA WorldCover: cropland\n"
        "AF_LANDCOVER_URBAN_CODE = 50     # ESA WorldCover: built-up\n\n\n"
        "def _fill_nan(arr, fallback=280.0):\n"
        "    if np.isnan(arr).any():\n"
        "        fill = np.nanmean(arr) if not np.isnan(arr).all() else fallback\n"
        "        arr = np.where(np.isnan(arr), fill, arr)\n"
        "    return arr\n\n\n"
        "def _local_mean_std(arr, win=21):\n"
        "    mean = uniform_filter(arr, size=win)\n"
        "    sq_mean = uniform_filter(arr ** 2, size=win)\n"
        "    std = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None)) + 1e-3\n"
        "    return mean, std\n\n\n"
        "def extract_af_features(chip, win=21):\n"
        "    \"\"\"chip = src.load_af(...). Возвращает (C, H, W) float32 в порядке AF_FEATURE_NAMES.\"\"\"\n"
        "    v, aux = chip['viirs'], chip['aux']\n"
        "    I1, I2, I3 = v['I1'], v['I2'], v['I3']\n"
        "    I4, I5 = _fill_nan(v['I4']), _fill_nan(v['I5'])\n"
        "    diff = I4 - I5\n\n"
        "    bg_I4_9, std_I4_9 = _local_mean_std(I4, 9)\n"
        "    bg_diff_9, std_diff_9 = _local_mean_std(diff, 9)\n"
        "    z_I4_w9 = (I4 - bg_I4_9) / std_I4_9\n"
        "    z_diff_w9 = (diff - bg_diff_9) / std_diff_9\n\n"
        "    bg_I4, std_I4 = _local_mean_std(I4, win)\n"
        "    bg_diff, std_diff = _local_mean_std(diff, win)\n"
        "    z_I4_w21 = (I4 - bg_I4) / std_I4\n"
        "    z_diff_w21 = (diff - bg_diff) / std_diff\n\n"
        "    bg_I4_41, std_I4_41 = _local_mean_std(I4, 41)\n"
        "    bg_diff_41, std_diff_41 = _local_mean_std(diff, 41)\n"
        "    z_I4_w41 = (I4 - bg_I4_41) / std_I4_41\n"
        "    z_diff_w41 = (diff - bg_diff_41) / std_diff_41\n"
        "    z_I4_scale_ratio = z_I4_w9 / (np.abs(z_I4_w41) + 1e-3)\n\n"
        "    solar_zenith = v['solar_zenith']\n"
        "    is_night = (solar_zenith > 90).astype(np.float32)\n"
        "    I1f = np.nan_to_num(I1, nan=0.0)\n"
        "    I2f = np.nan_to_num(I2, nan=0.0)\n"
        "    I3f = np.nan_to_num(I3, nan=0.0)\n"
        "    I1_I2_ratio = I1f / (I2f + 1e-3)\n"
        "    I2_I3_ratio = I2f / (I3f + 1e-3)\n"
        "    mean_refl_day = (I1f + I2f + I3f) / 3.0 * (1 - is_night)\n\n"
        "    landcover = aux['landcover']\n"
        "    is_cropland = (landcover == AF_LANDCOVER_CROPLAND_CODE).astype(np.float32)\n"
        "    is_urban = (landcover == AF_LANDCOVER_URBAN_CODE).astype(np.float32)\n\n"
        "    feats = np.stack([\n"
        "        I1, I2, I3, I4, I5, diff,\n"
        "        z_I4_w9, z_diff_w9,\n"
        "        z_I4_w21, z_diff_w21, bg_I4, bg_diff,\n"
        "        z_I4_w41, z_diff_w41, z_I4_scale_ratio,\n"
        "        solar_zenith, v['sensor_zenith'], is_night,\n"
        "        I1_I2_ratio, I2_I3_ratio, mean_refl_day,\n"
        "        landcover, aux['dem'], aux['t2m'], aux['rh2m'], aux['wind_speed'],\n"
        "        z_I4_w21 * is_cropland, z_diff_w21 * is_urban,\n"
        "    ], axis=0).astype(np.float32)\n"
        "    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)"
    ),
    md(
        "## 6. Обучающая таблица: позитивы + сложные негативы\n\n"
        "Полный пиксельный набор (~420 x 256 x 256 ≈ 27.5M строк) избыточен при ~9700 позитивных "
        "пикселях на весь train. Берём все позитивы + смесь «сложных» негативов (высокий z-score "
        "фона, то есть похожи на кандидата, но не помечены) и случайных — это стандартный hard "
        "negative mining для сильно несбалансированной пиксельной классификации."
    ),
    code(
        "def build_af_training_table(source, chip_ids, win=21, neg_per_pos=40, hard_neg_frac=0.5,\n"
        "                             seed=RANDOM_SEED, desc='building training table'):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    X_parts, y_parts = [], []\n"
        "    for chip_id in tqdm(list(chip_ids), desc=desc):\n"
        "        chip = source.load_af(chip_id)\n"
        "        feats = extract_af_features(chip, win=win)\n"
        "        mask = chip['mask']\n"
        "        valid = nodata_to_ignore(mask, chip['mask_nodata']) & (chip['viirs']['valid'] > 0)\n"
        "        pos_idx = np.flatnonzero((mask == 1) & valid)\n"
        "        neg_idx_all = np.flatnonzero((mask == 0) & valid)\n"
        "        if len(pos_idx) == 0 or len(neg_idx_all) == 0:\n"
        "            n_take = min(len(neg_idx_all), 200)\n"
        "            neg_idx = rng.choice(neg_idx_all, size=n_take, replace=False) if n_take > 0 else neg_idx_all\n"
        "        else:\n"
        "            z_I4_flat = feats[AF_FEATURE_NAMES.index('z_I4_w21')].ravel()[neg_idx_all]\n"
        "            n_hard = min(len(neg_idx_all), int(len(pos_idx) * neg_per_pos * hard_neg_frac))\n"
        "            hard_order = np.argsort(-z_I4_flat)[:n_hard]\n"
        "            hard_idx = neg_idx_all[hard_order]\n"
        "            remaining = np.setdiff1d(neg_idx_all, hard_idx)\n"
        "            n_rand = min(len(remaining), int(len(pos_idx) * neg_per_pos * (1 - hard_neg_frac)))\n"
        "            rand_idx = rng.choice(remaining, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)\n"
        "            neg_idx = np.concatenate([hard_idx, rand_idx]).astype(int)\n"
        "        idx = np.concatenate([pos_idx, neg_idx]).astype(int)\n"
        "        flat_feats = feats.reshape(feats.shape[0], -1)[:, idx].T\n"
        "        X_parts.append(flat_feats)\n"
        "        y_parts.append(mask.ravel()[idx])\n"
        "    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)\n\n\n"
        "def predict_af_proba(chip, model, win=21):\n"
        "    \"\"\"Инференс по ВСЕМ пикселям чипа - в отличие от build_af_training_table,\n"
        "    здесь нет сэмплирования/hard negative mining, это честное полное изображение.\n"
        "    Не трогает chip['mask'] - работает и на тестовых чипах, где маски нет.\n"
        "    Модель с objective='none' (focal loss) возвращает сырой логит - переводим в\n"
        "    вероятность вручную по флагу `_af_focal` (см. секцию 7/11)."
        "\"\"\"\n"
        "    feats = extract_af_features(chip, win=win)\n"
        "    flat = feats.reshape(feats.shape[0], -1).T\n"
        "    raw = model.predict(flat, num_iteration=model.best_iteration)\n"
        "    if getattr(model, '_af_focal', False):\n"
        "        raw = 1.0 / (1.0 + np.exp(-raw))\n"
        "    return raw.reshape(feats.shape[1:])\n\n\n"
        "def predict_af_proba_ensemble(chip, models, win=21):\n"
        "    \"\"\"Равновзвешенное усреднение вероятностей по ансамблю LightGBM - тот же приём,\n"
        "    что дал 1-е место xView2 (4 бэкбона, усреднённые без стекинга).\"\"\"\n"
        "    probs = [predict_af_proba(chip, m, win=win) for m in models]\n"
        "    return np.mean(probs, axis=0)\n\n\n"
        "# лёгкая проверка на 3 чипах, без обучения - убеждаемся, что фичи/таблица строятся без ошибок\n"
        "_X_smoke, _y_smoke = build_af_training_table(src, list(train_ids[:3]), desc='smoke test (3 чипа)')\n"
        "print('smoke test:', _X_smoke.shape, 'positives:', int(_y_smoke.sum()), 'NaN:', np.isnan(_X_smoke).sum())"
    ),
    md(
        "## 7. Обучение ансамбля LightGBM (тяжёлая ячейка — только на Colab/Kaggle)\n\n"
        "Вместо одной модели — **ансамбль из `AF_N_SEEDS` LightGBM** с разными seed/подвыборками "
        "признаков и бэггинга (равновзвешенное усреднение вероятностей — тот же приём, что "
        "принёс 1-е место xView2: 4 разных бэкбона, усреднённых без стекинга). Один из "
        "ансамбля обучается с **custom focal loss** вместо `is_unbalance=True` — разные "
        "objective дают разные ошибки, что увеличивает пользу от усреднения сильнее, чем "
        "просто разные seed одного и того же objective. `landcover` передаётся как "
        "категориальный признак.\n\n"
        "**Косметическая деталь лога, не баг**: у focal-модели (последний seed) "
        "`valid_0's binary_logloss` в логе обучения может уйти в отрицательные значения — "
        "встроенная метрика не в курсе, что objective кастомный, и не применяет sigmoid к "
        "сырым логитам перед подсчётом logloss. `auc` при этом честный (не зависит от "
        "монотонного преобразования), и `predict_af_proba`/`predict_af_proba_ensemble` сами "
        "применяют sigmoid к этой модели через флаг `_af_focal` — реальное качество смотрите "
        "по честному `F1_af` на полных изображениях чуть ниже, а не по этой метрике в логе.\n\n"
        "**Важно не перепутать метрики.** `valid_0's auc`/`binary_logloss`, которые печатает "
        "LightGBM во время обучения, считаются на **сэмплированной** таблице "
        "(`build_af_training_table` на `val_ids`) — там всего ~40 негативов на 1 позитив "
        "(`neg_per_pos=40`), а не реальное соотношение ~2830:1 на полном изображении. AUC там "
        "легко оказывается 0.998+ и почти ничего не говорит о реальном качестве — эта метрика "
        "нужна только механически, для ранней остановки. Единственная метрика, которой можно "
        "доверять, — `F1_af` на **полных** чипах, которую эта же ячейка честно считает сразу "
        "после обучения (при пороге 0.5, без подбора, для усреднённого ансамбля) и следующая "
        "ячейка (`sweep_af_threshold`) — уже с подобранным порогом."
    ),
    code(
        "def _lgb_progress_callback(total, desc='LightGBM boosting rounds'):\n"
        "    pbar = tqdm(total=total, desc=desc)\n\n"
        "    def _callback(env):\n"
        "        pbar.update(1)\n"
        "        if env.iteration + 1 >= env.end_iteration:\n"
        "            pbar.close()\n\n"
        "    _callback.order = 10\n"
        "    return _callback\n\n\n"
        "def focal_loss_lgb(y_pred, y_true, alpha=0.25, gamma=2.0, eps=1e-6):\n"
        "    \"\"\"grad/hess of binary focal loss wrt raw score, via central finite differences -\n"
        "    avoids hand-deriving (and risking a silent error in) the analytic formula.\"\"\"\n"
        "    def _value(x, t):\n"
        "        p = 1.0 / (1.0 + np.exp(-x))\n"
        "        p = np.clip(p, 1e-7, 1 - 1e-7)\n"
        "        pt = np.where(t == 1, p, 1 - p)\n"
        "        alpha_t = np.where(t == 1, alpha, 1 - alpha)\n"
        "        return -alpha_t * (1 - pt) ** gamma * np.log(pt)\n\n"
        "    grad = (_value(y_pred + eps, y_true) - _value(y_pred - eps, y_true)) / (2 * eps)\n"
        "    hess = (_value(y_pred + eps, y_true) - 2 * _value(y_pred, y_true) + _value(y_pred - eps, y_true)) / (eps ** 2)\n"
        "    return grad, np.clip(hess, 1e-6, None)\n\n\n"
        "AF_N_SEEDS = 3  # размер ансамбля LightGBM - время обучения не ограничено, ансамбль почти всегда лучше одной модели\n\n"
        "if RUN_HEAVY_TRAINING:\n"
        "    import json\n"
        "    import time\n\n"
        "    import lightgbm as lgb\n\n"
        "    X_train, y_train = build_af_training_table(src, train_ids, desc='training table (train)')\n"
        "    X_val_tab, y_val_tab = build_af_training_table(src, val_ids, desc='training table (val)')\n\n"
        "    af_models = []\n"
        "    af_model_paths = []\n"
        "    NUM_BOOST_ROUND = 2000\n"
        "    for seed_i in range(AF_N_SEEDS):\n"
        "        use_focal = seed_i == AF_N_SEEDS - 1  # последний в ансамбле - другой objective для разнообразия ошибок\n"
        "        seed = RANDOM_SEED + seed_i\n"
        "        train_set = lgb.Dataset(X_train, label=y_train, feature_name=AF_FEATURE_NAMES,\n"
        "                                 categorical_feature=[AF_LANDCOVER_IDX])\n"
        "        val_set = lgb.Dataset(X_val_tab, label=y_val_tab, feature_name=AF_FEATURE_NAMES,\n"
        "                               categorical_feature=[AF_LANDCOVER_IDX], reference=train_set)\n"
        "        params = dict(\n"
        "            metric=['auc', 'binary_logloss'],\n"
        "            num_leaves=63, learning_rate=0.05,\n"
        "            feature_fraction=0.8 + 0.05 * (seed_i % 3), bagging_fraction=0.8 + 0.05 * (seed_i % 2),\n"
        "            bagging_freq=1, min_data_in_leaf=50, seed=seed, verbose=-1,\n"
        "        )\n"
        "        if use_focal:\n"
        "            # актуальный API LightGBM (4.x убрал отдельный аргумент fobj= у train()) -\n"
        "            # кастомный objective передаётся как сам callable в params['objective']\n"
        "            params['objective'] = lambda preds, data: focal_loss_lgb(preds, data.get_label())\n"
        "        else:\n"
        "            params['objective'] = 'binary'\n"
        "            params['is_unbalance'] = True\n"
        "        model_i = lgb.train(\n"
        "            params, train_set, num_boost_round=NUM_BOOST_ROUND, valid_sets=[val_set],\n"
        "            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200),\n"
        "                       _lgb_progress_callback(NUM_BOOST_ROUND, desc=f'LightGBM seed {seed_i+1}/{AF_N_SEEDS}')],\n"
        "        )\n"
        "        if use_focal:\n"
        "            # focal-модель предсказывает сырой логит (объектив 'none'), приводим к вероятности вручную\n"
        "            model_i._af_focal = True\n"
        "        model_path = os.path.join(CHECKPOINT_DIR, f'af_lightgbm_{seed_i}.txt')\n"
        "        model_i.save_model(model_path)\n"
        "        af_models.append(model_i)\n"
        "        af_model_paths.append(model_path)\n"
        "        print(f'seed {seed_i+1}/{AF_N_SEEDS} (focal={use_focal}) best_iteration={model_i.best_iteration}')\n\n"
        "    af_meta = {\n"
        "        'feature_names': AF_FEATURE_NAMES,\n"
        "        'categorical_feature_idx': [AF_LANDCOVER_IDX],\n"
        "        'n_seeds': AF_N_SEEDS,\n"
        "        'focal_model_idx': AF_N_SEEDS - 1,\n"
        "        'best_iterations': [m.best_iteration for m in af_models],\n"
        "        'seed': RANDOM_SEED,\n"
        "        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),\n"
        "    }\n"
        "    af_meta_path = os.path.join(CHECKPOINT_DIR, 'af_lightgbm_meta.json')\n"
        "    with open(af_meta_path, 'w', encoding='utf-8') as f:\n"
        "        json.dump(af_meta, f, ensure_ascii=False, indent=2)\n"
        "    print('Сохранено:', af_model_paths, '|', af_meta_path)\n\n"
        "    # проверка воспроизводимости для модели 0 (остальные - тот же механизм)\n"
        "    _af_reloaded = lgb.Booster(model_file=af_model_paths[0])\n"
        "    _check_a = af_models[0].predict(X_val_tab[:200], num_iteration=af_models[0].best_iteration)\n"
        "    _check_b = _af_reloaded.predict(X_val_tab[:200], num_iteration=af_models[0].best_iteration)\n"
        "    assert np.allclose(_check_a, _check_b), 'сохранённая модель не воспроизводит предсказания!'\n"
        "    print('Проверка воспроизводимости пройдена: предсказания модели с диска совпадают.')\n\n"
        "    # ЧЕСТНАЯ проверка ансамбля на полных изображениях (порог 0.5, до подбора) - это F1_af,\n"
        "    # а не auc/logloss выше, которые считались на сэмплированной таблице\n"
        "    _quick_acc = ConfusionAccumulator(classes=(1,))\n"
        "    for _chip in tqdm(iter_chips(src, 'af', val_ids), total=len(val_ids), desc='честная проверка (полные чипы)'):\n"
        "        _proba = predict_af_proba_ensemble(_chip, af_models)\n"
        "        _valid = nodata_to_ignore(_chip['mask'], _chip['mask_nodata'])\n"
        "        _quick_acc.update((_proba >= 0.5).astype(np.uint8), _chip['mask'], valid=_valid)\n"
        "    print('F1_af ансамбля на ПОЛНЫХ изображениях, порог 0.5 (сравнивать с baseline, не auc выше):',\n"
        "          f1_af(_quick_acc))\n"
        "    print('Порог 0.5 почти наверняка не оптимален - подбор в следующей ячейке.')\n"
        "else:\n"
        "    print('RUN_HEAVY_TRAINING=False - обучение LightGBM пропущено (Colab/Kaggle).')\n"
        "    af_models = None"
    ),
    md(
        "## 8. CNN как второй независимый классификатор — опционально (`AF_TRAIN_CNN`)\n\n"
        "**По умолчанию выключена.** Идея была такая: LightGBM смотрит на каждый пиксель "
        "по отдельности, а свёрточная сеть на том же 28-канальном стеке видит форму и "
        "вытянутость горячей области (Pereira et al. 2021 показали, что такая комбинация "
        "обходит каждую модель по отдельности). Но **на реальном прогоне это не сработало**: "
        "сама CNN дала `F1_af ≈ 0.41` против `≈0.96` у LightGBM-ансамбля, и max-блендинг с "
        "настолько более слабой моделью только добавлял ложных срабатываний — итоговый F1 "
        "падал до `≈0.67`. Секция 10 это ловит автоматически (честно сравнивает обе "
        "стратегии и берёт лучшую), но раз исход известен — не тратим ~5 минут обучения "
        "впустую. Поставьте `AF_TRAIN_CNN = True`, если хотите перепроверить самостоятельно."
    ),
    code(
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch.utils.data import Dataset, DataLoader\n\n\n"
        "class ConvBlock(nn.Module):\n"
        "    def __init__(self, cin, cout):\n"
        "        super().__init__()\n"
        "        self.block = nn.Sequential(\n"
        "            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),\n"
        "            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        return self.block(x)\n\n\n"
        "class AFCNN(nn.Module):\n"
        "    \"\"\"Компактный U-Net для бинарной сегментации огня - второе, "
        "\\\"пространственное\\\" мнение поверх попиксельного LightGBM.\"\"\"\n\n"
        "    def __init__(self, in_channels, base=16):\n"
        "        super().__init__()\n"
        "        self.input_norm = nn.BatchNorm2d(in_channels, affine=False)\n"
        "        self.enc1 = ConvBlock(in_channels, base)\n"
        "        self.enc2 = ConvBlock(base, base * 2)\n"
        "        self.enc3 = ConvBlock(base * 2, base * 4)\n"
        "        self.pool = nn.MaxPool2d(2)\n"
        "        self.bottleneck = ConvBlock(base * 4, base * 8)\n"
        "        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)\n"
        "        self.dec3 = ConvBlock(base * 8, base * 4)\n"
        "        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)\n"
        "        self.dec2 = ConvBlock(base * 4, base * 2)\n"
        "        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)\n"
        "        self.dec1 = ConvBlock(base * 2, base)\n"
        "        self.out_conv = nn.Conv2d(base, 1, 1)\n\n"
        "    def forward(self, x):\n"
        "        x = self.input_norm(x)\n"
        "        e1 = self.enc1(x)\n"
        "        e2 = self.enc2(self.pool(e1))\n"
        "        e3 = self.enc3(self.pool(e2))\n"
        "        b = self.bottleneck(self.pool(e3))\n"
        "        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))\n"
        "        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))\n"
        "        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))\n"
        "        return self.out_conv(d1).squeeze(1)  # (N, H, W) логиты\n\n\n"
        "class AFBCEDiceLoss(nn.Module):\n"
        "    \"\"\"BCE (pos_weight под дисбаланс) + soft Dice, только на valid-пикселях - "
        "бинарный аналог DiceCELoss из модуля BS.\"\"\"\n\n"
        "    def __init__(self, pos_weight):\n"
        "        super().__init__()\n"
        "        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))\n\n"
        "    def forward(self, logits, target, valid):\n"
        "        bce = F.binary_cross_entropy_with_logits(\n"
        "            logits, target, reduction='none', pos_weight=self.pos_weight,\n"
        "        )\n"
        "        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)\n"
        "        probs = torch.sigmoid(logits)\n"
        "        inter = (probs * target * valid).sum()\n"
        "        union = (probs * valid).sum() + (target * valid).sum()\n"
        "        dice = (2 * inter + 1.0) / (union + 1.0)\n"
        "        return bce + (1.0 - dice)\n\n\n"
        "class AFChipDataset(Dataset):\n"
        "    def __init__(self, source, chip_ids, augment=False):\n"
        "        self.source = source\n"
        "        self.chip_ids = list(chip_ids)\n"
        "        self.augment = augment\n\n"
        "    def __len__(self):\n"
        "        return len(self.chip_ids)\n\n"
        "    def __getitem__(self, i):\n"
        "        chip = self.source.load_af(self.chip_ids[i])\n"
        "        feats = extract_af_features(chip)\n"
        "        mask = chip['mask'].astype(np.float32)\n"
        "        valid = nodata_to_ignore(mask, chip['mask_nodata']) & (chip['viirs']['valid'] > 0)\n"
        "        if self.augment:\n"
        "            if np.random.rand() < 0.5:\n"
        "                feats, mask, valid = feats[:, :, ::-1].copy(), mask[:, ::-1].copy(), valid[:, ::-1].copy()\n"
        "            if np.random.rand() < 0.5:\n"
        "                feats, mask, valid = feats[:, ::-1, :].copy(), mask[::-1, :].copy(), valid[::-1, :].copy()\n"
        "            k = np.random.randint(0, 4)\n"
        "            feats = np.ascontiguousarray(np.rot90(feats, k, axes=(1, 2)))\n"
        "            mask = np.ascontiguousarray(np.rot90(mask, k))\n"
        "            valid = np.ascontiguousarray(np.rot90(valid, k))\n"
        "        return (\n"
        "            torch.from_numpy(feats.copy()),\n"
        "            torch.from_numpy(mask.copy()),\n"
        "            torch.from_numpy(valid.astype(np.float32).copy()),\n"
        "        )\n\n\n"
        "# Быстрая проверка кода модели и лосса (НЕ обучение, ~1 сек на CPU)\n"
        "_af_smoke_model = AFCNN(in_channels=len(AF_FEATURE_NAMES), base=4)\n"
        "_af_dummy_feats = torch.randn(2, len(AF_FEATURE_NAMES), 32, 32)\n"
        "_af_dummy_mask = torch.randint(0, 2, (2, 32, 32)).float()\n"
        "_af_dummy_valid = torch.ones(2, 32, 32)\n"
        "_af_logits = _af_smoke_model(_af_dummy_feats)\n"
        "_af_loss = AFBCEDiceLoss(pos_weight=10.0)(_af_logits, _af_dummy_mask, _af_dummy_valid)\n"
        "_af_loss.backward()\n"
        "print('smoke test OK, logits shape:', tuple(_af_logits.shape), ' loss:', float(_af_loss.detach()))\n\n\n"
        "def train_af_cnn(source, train_ids, val_ids, pos_weight, epochs=15, batch_size=8, lr=1e-3, device=DEVICE):\n"
        "    train_ds = AFChipDataset(source, train_ids, augment=True)\n"
        "    val_ds = AFChipDataset(source, val_ids, augment=False)\n"
        "    # num_workers=0: с >0 на Python 3.13/Colab worker-процессы иногда падают с AssertionError\n"
        "    # при завершении (безобидно, но захламляет лог) - на наших объёмах данных не критично.\n"
        "    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)\n\n"
        "    model = AFCNN(in_channels=len(AF_FEATURE_NAMES)).to(device)\n"
        "    criterion = AFBCEDiceLoss(pos_weight=pos_weight).to(device)\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)\n"
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)\n\n"
        "    best_f1, best_state = -1.0, None\n"
        "    epoch_bar = tqdm(range(epochs), desc='AF CNN epochs')\n"
        "    for epoch in epoch_bar:\n"
        "        model.train()\n"
        "        running = 0.0\n"
        "        batch_bar = tqdm(train_loader, desc=f'epoch {epoch+1}/{epochs} (train)', leave=False)\n"
        "        for feats, mask, valid in batch_bar:\n"
        "            feats, mask, valid = feats.to(device), mask.to(device), valid.to(device)\n"
        "            optimizer.zero_grad()\n"
        "            loss = criterion(model(feats), mask, valid)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            running += loss.item() * feats.size(0)\n"
        "            batch_bar.set_postfix(loss=f'{loss.item():.4f}')\n"
        "        scheduler.step()\n\n"
        "        model.eval()\n"
        "        acc_cnn = ConfusionAccumulator(classes=(1,))\n"
        "        with torch.no_grad():\n"
        "            for idx in tqdm(range(len(val_ds)), desc=f'epoch {epoch+1}/{epochs} (val)', leave=False):\n"
        "                feats, mask, valid = val_ds[idx]\n"
        "                proba = torch.sigmoid(model(feats.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()\n"
        "                acc_cnn.update((proba >= 0.5).astype(np.uint8), mask.numpy().astype(np.uint8),\n"
        "                                valid=valid.numpy().astype(bool))\n"
        "        val_f1 = f1_af(acc_cnn)\n"
        "        epoch_bar.set_postfix(loss=f'{running/len(train_ds):.4f}', F1_af=f'{val_f1:.3f}')\n"
        "        if val_f1 > best_f1:\n"
        "            best_f1 = val_f1\n"
        "            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}\n"
        "            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'af_cnn_best.pt'))\n"
        "    model.load_state_dict(best_state)\n"
        "    return model, best_f1\n\n\n"
        "AF_TRAIN_CNN = False  # см. markdown выше: на реальном прогоне CNN проиграла и была отвергнута\n\n"
        "if RUN_HEAVY_TRAINING and AF_TRAIN_CNN and af_models is not None:\n"
        "    _train_meta = meta_af[meta_af['chip_id'].isin(train_ids)]\n"
        "    _pos_px = _train_meta['n_fire_px'].fillna(0).sum()\n"
        "    _total_px = len(_train_meta) * 256 * 256\n"
        "    af_cnn_pos_weight = float((_total_px - _pos_px) / max(_pos_px, 1))\n"
        "    print('AF CNN pos_weight:', af_cnn_pos_weight)\n"
        "    af_cnn_model, af_cnn_val_f1 = train_af_cnn(src, train_ids, val_ids, af_cnn_pos_weight)\n"
        "    print('AF CNN отдельно (без блендинга), F1_af (val, порог 0.5):', af_cnn_val_f1)\n"
        "else:\n"
        "    print('AF_TRAIN_CNN=False (или RUN_HEAVY_TRAINING=False) - CNN не обучается, '\n"
        "          'работаем на LightGBM-ансамбле.')\n"
        "    af_cnn_model = None"
    ),
    md(
        "## 9. Max-блендинг LightGBM-ансамбля и CNN\n\n"
        "1-е место DrivenData STAC Overflow комбинировало сегментационную сеть с независимым "
        "табличным подходом через **поэлементный max**, а не среднее — так ловятся пиксели, "
        "которые каждая модель по отдельности упускает. При таком дисбалансе (~9700 позитивов "
        "на 27.5M пикселей) F1 recall-голоден, и max — почти бесплатный способ выменять часть "
        "precision на recall."
    ),
    code(
        "def predict_af_proba_blend(chip, models, cnn_model, win=21, device=DEVICE):\n"
        "    p_lgb = predict_af_proba_ensemble(chip, models, win=win)\n"
        "    if cnn_model is None:\n"
        "        return p_lgb\n"
        "    feats = extract_af_features(chip, win=win)\n"
        "    with torch.no_grad():\n"
        "        p_cnn = torch.sigmoid(\n"
        "            cnn_model(torch.from_numpy(feats).unsqueeze(0).to(device))\n"
        "        ).squeeze(0).cpu().numpy()\n"
        "    return np.maximum(p_lgb, p_cnn)"
    ),
    md(
        "## 10. Подбор порога под F1_af (+ калибровка по landcover)\n\n"
        "Порог бинаризации подбирается на валидации максимизацией `F1_af` (микро-усреднение по "
        "полным изображениям), а не accuracy/logloss/auc с шага 7. Дополнительно — **отдельный "
        "порог для каждого кода landcover**: раз ложные срабатывания кучкуются на пашне/в "
        "городе, а `landcover` уже есть как признак, слегка более высокий порог именно для "
        "этих классов выжимает остаточные ошибки без переобучения.\n\n"
        "**Важно (реальная находка на первом прогоне)**: max-блендинг с CNN — не гарантированно "
        "лучше чистого LightGBM-ансамбля. В одном из прогонов CNN оказался намного слабее "
        "(F1_af отдельно ≈0.41 против ≈0.96 у LightGBM-ансамбля), и `max()` с намного более "
        "слабой моделью только добавлял ложных срабатываний — после агрессивного повышения "
        "порога итоговый F1 всё равно просел до ≈0.67 против ≈0.96 у одного LightGBM-ансамбля. "
        "Поэтому ниже **обе стратегии** (чистый ансамбль и блендинг с CNN) честно сравниваются "
        "на валидации, и используется та, что реально лучше — а не блендинг по умолчанию. "
        "Вероятности кэшируются один раз на каждую стратегию — иначе пересчёт для 19 порогов "
        "был бы намного медленнее."
    ),
    code(
        "def sweep_af_threshold(source, chip_ids, proba_fn, thresholds=np.linspace(0.05, 0.95, 19)):\n"
        "    chip_ids = list(chip_ids)\n"
        "    cached = []\n"
        "    for chip in tqdm(iter_chips(source, 'af', chip_ids), total=len(chip_ids),\n"
        "                      desc='threshold sweep: кэшируем вероятности'):\n"
        "        proba = proba_fn(chip)\n"
        "        valid = nodata_to_ignore(chip['mask'], chip['mask_nodata'])\n"
        "        cached.append((proba, chip['mask'], valid, chip['aux']['landcover']))\n\n"
        "    best_t, best_f1 = 0.5, -1.0\n"
        "    for t in tqdm(thresholds, desc='threshold sweep: глобальный порог'):\n"
        "        acc_t = ConfusionAccumulator(classes=(1,))\n"
        "        for proba, mask, valid, _ in cached:\n"
        "            acc_t.update((proba >= t).astype(np.uint8), mask, valid=valid)\n"
        "        f1 = f1_af(acc_t)\n"
        "        if f1 > best_f1:\n"
        "            best_f1, best_t = f1, t\n"
        "    return best_t, best_f1, cached\n\n\n"
        "def calibrate_af_thresholds_by_landcover(cached, global_threshold, thresholds=np.linspace(0.05, 0.95, 19)):\n"
        "    landcover_codes = np.unique(np.concatenate([lc.ravel() for _, _, _, lc in cached]))\n"
        "    landcover_codes = landcover_codes[~np.isnan(landcover_codes)]  # AUX landcover - float32, бывает NaN на краях покрытия\n"
        "    per_landcover = {}\n"
        "    for code in landcover_codes:\n"
        "        best_t, best_f1 = float(global_threshold), -1.0\n"
        "        for t in thresholds:\n"
        "            acc_t = ConfusionAccumulator(classes=(1,))\n"
        "            for proba, mask, valid, lc in cached:\n"
        "                sel = valid & (lc == code)\n"
        "                if not sel.any():\n"
        "                    continue\n"
        "                acc_t.update((proba >= t).astype(np.uint8), mask, valid=sel)\n"
        "            f1 = f1_af(acc_t)\n"
        "            if f1 > best_f1:\n"
        "                best_f1, best_t = f1, t\n"
        "        per_landcover[int(code)] = float(best_t)\n"
        "    return per_landcover\n\n\n"
        "def apply_af_thresholds(proba, landcover, global_threshold, per_landcover_thresholds=None):\n"
        "    if not per_landcover_thresholds:\n"
        "        return proba >= global_threshold\n"
        "    thr_map = np.full(landcover.shape, global_threshold, dtype=np.float32)\n"
        "    for code, t in per_landcover_thresholds.items():\n"
        "        thr_map[landcover == int(code)] = t\n"
        "    return proba >= thr_map\n\n\n"
        "if RUN_HEAVY_TRAINING and af_models is not None:\n"
        "    def _af_lgb_only_fn(chip):\n"
        "        return predict_af_proba_blend(chip, af_models, None)\n\n"
        "    def _af_blend_fn(chip):\n"
        "        return predict_af_proba_blend(chip, af_models, af_cnn_model)\n\n"
        "    _t_lgb, _f1_lgb, _cache_lgb = sweep_af_threshold(src, val_ids, _af_lgb_only_fn)\n"
        "    print('LightGBM-ансамбль без CNN: порог=', _t_lgb, ' F1_af(val)=', _f1_lgb)\n\n"
        "    if af_cnn_model is not None:\n"
        "        _t_blend, _f1_blend, _cache_blend = sweep_af_threshold(src, val_ids, _af_blend_fn)\n"
        "        print('Max-блендинг с CNN:        порог=', _t_blend, ' F1_af(val)=', _f1_blend)\n"
        "    else:\n"
        "        _f1_blend = -1.0\n\n"
        "    if af_cnn_model is not None and _f1_blend >= _f1_lgb:\n"
        "        use_cnn_in_blend = True\n"
        "        global_threshold, best_f1_val, _af_val_cache = _t_blend, _f1_blend, _cache_blend\n"
        "        print('Выбран max-блендинг с CNN (лучше или наравне на валидации)')\n"
        "    else:\n"
        "        use_cnn_in_blend = False\n"
        "        global_threshold, best_f1_val, _af_val_cache = _t_lgb, _f1_lgb, _cache_lgb\n"
        "        print('Выбран LightGBM-ансамбль БЕЗ CNN (блендинг не улучшил или ухудшил F1_af на валидации)')\n\n"
        "    per_landcover_thresholds = calibrate_af_thresholds_by_landcover(_af_val_cache, global_threshold)\n"
        "    print('глобальный порог:', global_threshold, ' F1_af (val, выбранная стратегия):', best_f1_val,\n"
        "          ' vs baseline:', f1_af(acc))\n"
        "    print('пороги по landcover:', per_landcover_thresholds)\n\n"
        "    af_meta_path = os.path.join(CHECKPOINT_DIR, 'af_lightgbm_meta.json')\n"
        "    with open(af_meta_path, encoding='utf-8') as f:\n"
        "        af_meta = json.load(f)\n"
        "    af_meta['global_threshold'] = float(global_threshold)\n"
        "    af_meta['per_landcover_thresholds'] = per_landcover_thresholds\n"
        "    af_meta['f1_af_val_best'] = float(best_f1_val)\n"
        "    af_meta['use_cnn_in_blend'] = use_cnn_in_blend\n"
        "    with open(af_meta_path, 'w', encoding='utf-8') as f:\n"
        "        json.dump(af_meta, f, ensure_ascii=False, indent=2)\n"
        "    print('Пороги и метрика добавлены в', af_meta_path)"
    ),
    md(
        "## 11. Постобработка: фильтрация связных компонент по уверенности\n\n"
        "Плоское правило «убрать блобы < K пикселей» рискует стереть настоящие мелкие очаги "
        "(медиана позитивного чипа — всего ~21 пиксель). Безопаснее сделать эффективный порог "
        "функцией и вероятности, и размера блоба: убираем маленький блоб, только если он ещё и "
        "низкоуверенный, а одиночный высокоуверенный пиксель пропускаем без изменений."
    ),
    code(
        "from scipy import ndimage as ndi\n\n\n"
        "def confidence_tiered_filter(proba, pred_mask, high_conf=0.9, min_blob_size=3):\n"
        "    labeled, n = ndi.label(pred_mask)\n"
        "    keep = np.zeros_like(pred_mask, dtype=bool)\n"
        "    for i in range(1, n + 1):\n"
        "        blob = labeled == i\n"
        "        if blob.sum() >= min_blob_size or proba[blob].max() >= high_conf:\n"
        "            keep |= blob\n"
        "    return keep"
    ),
    md(
        "## 12. Быстрый baseline-сабмишн (без обучения, без GPU)\n\n"
        "Не обязательно ждать переобучения модели, чтобы что-то сдать: контекстный "
        "baseline из шага 2 не требует ни GPU, ни LightGBM — можно прогнать по тесту "
        "прямо сейчас, за секунды, и получить (а) точку отсчёта на реальном лидерборде "
        "и (б) проверку, что формат сдачи/механика сайта организаторов вообще работает, "
        "пока основная модель дообучается на Colab."
    ),
    code(
        "_bl_test_source = resolve_test_source(DATASET_DIR)\n"
        "_bl_sample_submission = _bl_test_source.read_csv('sample_submission.csv')\n"
        "_bl_af_ids = _bl_sample_submission.loc[\n"
        "    _bl_sample_submission['chip_id'].str.startswith('AF_'), 'chip_id'\n"
        "].unique()\n\n"
        "_baseline_rows = []\n"
        "for chip_id in tqdm(_bl_af_ids, desc='baseline-инференс AF на тесте'):\n"
        "    chip = _bl_test_source.load_af(chip_id)\n"
        "    I4c = np.nan_to_num(chip['viirs']['I4'], nan=280.0)\n"
        "    I5c = np.nan_to_num(chip['viirs']['I5'], nan=280.0)\n"
        "    pred = baseline_af_predict(I4c, I5c)\n"
        "    _baseline_rows.append(submission_row(chip_id, 1, pred.astype(bool)))\n\n"
        "submission_af_baseline = pd.DataFrame(_baseline_rows)\n"
        "_baseline_path = os.path.join(CHECKPOINT_DIR, 'submission_af_baseline.csv')\n"
        "submission_af_baseline.to_csv(_baseline_path, index=False)\n"
        "_bl_af_sample = _bl_sample_submission[_bl_sample_submission['chip_id'].isin(_bl_af_ids)]\n"
        "print('Baseline-сабмишн AF сохранён:', _baseline_path, submission_af_baseline.shape)\n"
        "print('Проблемы валидации (пусто = ок):', validate_submission(submission_af_baseline, _bl_af_sample))\n"
        "_bl_test_source.close()"
    ),
    md(
        "## 13. Инференс на тесте (ансамбль ± CNN + пороги + постобработка) + RLE для submission.csv\n\n"
        "Тест уже выложен (`fire-test-renamed`, 180 AF-чипов, без масок). Для AF-чипа — ровно "
        "одна строка (`class_id=1`). Полный пайплайн: LightGBM-ансамбль (± CNN — какая "
        "стратегия победила на валидации в секции 10, `use_cnn_in_blend`) → порог (глобальный "
        "или по landcover) → фильтрация связных компонент по уверенности. Если ничего не "
        "обучалось в этой сессии (`RUN_HEAVY_TRAINING=False`) — все компоненты (ансамбль, CNN, "
        "пороги, выбранная стратегия) подхватываются из `CHECKPOINT_DIR`, так что инференс "
        "можно гонять отдельно от обучения."
    ),
    code(
        "def predict_and_encode_af(chip_id, models, cnn_model, global_threshold,\n"
        "                           per_landcover_thresholds, source):\n"
        "    chip = source.load_af(chip_id)\n"
        "    proba = predict_af_proba_blend(chip, models, cnn_model)\n"
        "    pred_mask = apply_af_thresholds(proba, chip['aux']['landcover'],\n"
        "                                     global_threshold, per_landcover_thresholds)\n"
        "    pred_mask = confidence_tiered_filter(proba, pred_mask)\n"
        "    return submission_row(chip_id, 1, pred_mask)\n\n\n"
        "if af_models is None:\n"
        "    _af_meta_path = os.path.join(CHECKPOINT_DIR, 'af_lightgbm_meta.json')\n"
        "    if os.path.exists(_af_meta_path):\n"
        "        import json\n"
        "        import lightgbm as lgb\n"
        "        with open(_af_meta_path, encoding='utf-8') as f:\n"
        "            af_meta = json.load(f)\n"
        "        af_models = []\n"
        "        for i in range(af_meta['n_seeds']):\n"
        "            m = lgb.Booster(model_file=os.path.join(CHECKPOINT_DIR, f'af_lightgbm_{i}.txt'))\n"
        "            # у Booster, загруженного с диска, нет своего best_iteration - берём из метаданных\n"
        "            m.best_iteration = af_meta['best_iterations'][i]\n"
        "            if i == af_meta.get('focal_model_idx'):\n"
        "                m._af_focal = True\n"
        "            af_models.append(m)\n"
        "        global_threshold = af_meta.get('global_threshold', 0.5)\n"
        "        per_landcover_thresholds = af_meta.get('per_landcover_thresholds')\n"
        "        use_cnn_in_blend = af_meta.get('use_cnn_in_blend', False)\n"
        "        _af_cnn_ckpt = os.path.join(CHECKPOINT_DIR, 'af_cnn_best.pt')\n"
        "        if use_cnn_in_blend and os.path.exists(_af_cnn_ckpt):\n"
        "            af_cnn_model = AFCNN(in_channels=len(AF_FEATURE_NAMES)).to(DEVICE)\n"
        "            af_cnn_model.load_state_dict(torch.load(_af_cnn_ckpt, map_location=DEVICE))\n"
        "            af_cnn_model.eval()\n"
        "        else:\n"
        "            af_cnn_model = None\n"
        "        print('Ансамбль (', len(af_models), 'моделей) подхвачен из чекпоинта, порог:', global_threshold,\n"
        "              '| CNN в блендинге:', af_cnn_model is not None)\n"
        "    else:\n"
        "        print('Нет ни обученной в этой сессии модели, ни чекпоинта в CHECKPOINT_DIR - '\n"
        "              'инференс на тесте пропущен. Поставьте RUN_HEAVY_TRAINING=True и обучите модель.')\n\n"
        "if af_models is not None:\n"
        "    test_source = resolve_test_source(DATASET_DIR)\n"
        "    sample_submission = test_source.read_csv('sample_submission.csv')\n"
        "    af_test_ids = sample_submission.loc[\n"
        "        sample_submission['chip_id'].str.startswith('AF_'), 'chip_id'\n"
        "    ].unique()\n\n"
        "    _af_cnn_for_inference = af_cnn_model if use_cnn_in_blend else None\n"
        "    af_rows = [\n"
        "        predict_and_encode_af(chip_id, af_models, _af_cnn_for_inference, global_threshold,\n"
        "                               per_landcover_thresholds, test_source)\n"
        "        for chip_id in tqdm(af_test_ids, desc='инференс AF на тесте')\n"
        "    ]\n"
        "    submission_af = pd.DataFrame(af_rows)\n"
        "    af_sub_path = os.path.join(CHECKPOINT_DIR, 'submission_af.csv')\n"
        "    submission_af.to_csv(af_sub_path, index=False)\n\n"
        "    af_sample = sample_submission[sample_submission['chip_id'].isin(af_test_ids)]\n"
        "    problems = validate_submission(submission_af, af_sample)\n"
        "    print('Сохранено:', af_sub_path, submission_af.shape)\n"
        "    print('Проблемы валидации (пусто = ок):', problems)\n"
        "    test_source.close()"
    ),
    code("src.close()"),
]

with open("01_af_binary_fire_detection.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb(af_cells), f)


# ---------------------------------------------------------------------------
# Notebook 2: BS module - burn contour + severity classification
# ---------------------------------------------------------------------------

bs_cells = [
    md(
        "# Модуль 2 — контур гари и степень поражения (BS)\n\n"
        "Вход: BS-чип 512x512 @ 20 м — Sentinel-2 pre/post (10 каналов: B2,B3,B4,B5,B6,B7,B8A,B11,B12,SCL), "
        "Sentinel-1 pre/post (VV,VH), AUX (dem, slope, landcover).\n\n"
        "Выход: маска 0-3 (0 = не горело, 1/2/3 = слабая/средняя/сильная степень поражения).\n\n"
        "Метрика: `IoU_burn` (бинарный контур гари, вес 0.35) + `mIoU_sev` (среднее IoU по классам "
        "1/2/3, вес 0.30).\n\n"
        "Модель: U-Net (обучается с нуля, вход — 33-канальный стек Sentinel-2/1 + спектральные индексы "
        "+ AUX + облачная маска), обучение на Colab/Kaggle с GPU."
    ),
    md("## 0. Настройка"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from tqdm.auto import tqdm\n\n"
        "RANDOM_SEED = 42\n"
        "np.random.seed(RANDOM_SEED)"
    ),
    md(
        "Ноутбук самодостаточный — код `src/dataio.py`, `src/rle.py`, `src/metrics.py` встроен "
        "прямо сюда (см. следующие 3 ячейки), поэтому запуск в Colab работает при загрузке "
        "одного `.ipynb`-файла, без остального репозитория рядом."
    ),
    embed_src_cell("dataio.py"),
    embed_src_cell("rle.py"),
    embed_src_cell("metrics.py"),
    embed_src_cell("bs_pipeline.py"),
    md(KAGGLE_MD),
    code(KAGGLE_CODE),
    md(HEAVY_GUARD_MD),
    code(HEAVY_GUARD_CODE),
    md(CHECKPOINT_MD),
    code(CHECKPOINT_CODE),
    code(
        "src = resolve_train_source(DATASET_DIR)\n"
        "meta_bs = src.meta('bs')\n"
        "meta_bs.head()"
    ),
    md(
        "## 1. EDA\n\n"
        "- распределение классов степени поражения (`sev1_px`/`sev2_px`/`sev3_px`);\n"
        "- облачность `cloud_frac`;\n"
        "- визуальная проверка pre/post RGB + маски."
    ),
    code(
        "sev_cols = ['sev1_px', 'sev2_px', 'sev3_px']\n"
        "totals = meta_bs[sev_cols].sum()\n"
        "print(totals)\n"
        "print('доли:', (totals / totals.sum()).round(3).to_dict())\n"
        "meta_bs['cloud_frac'].hist(bins=30)\n"
        "plt.xlabel('cloud_frac'); plt.ylabel('число чипов'); plt.title('Облачность по чипам')"
    ),
    code(
        "chip_id = meta_bs.loc[meta_bs['sev2_px'] > 1000, 'chip_id'].iloc[0]\n"
        "sample = src.load_bs(chip_id)\n"
        "rgb_pre = np.stack([sample['s2_pre']['B4'], sample['s2_pre']['B3'], sample['s2_pre']['B2']], axis=-1)\n"
        "rgb_post = np.stack([sample['s2_post']['B4'], sample['s2_post']['B3'], sample['s2_post']['B2']], axis=-1)\n"
        "\n"
        "def stretch(img, p=2):\n"
        "    lo, hi = np.percentile(img, p), np.percentile(img, 100 - p)\n"
        "    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
        "axes[0].imshow(stretch(rgb_pre)); axes[0].set_title('pre (RGB)')\n"
        "axes[1].imshow(stretch(rgb_post)); axes[1].set_title('post (RGB)')\n"
        "axes[2].imshow(sample['mask'], cmap='YlOrRd', vmin=0, vmax=3); axes[2].set_title(f'маска, {chip_id}')\n"
        "for ax in axes: ax.axis('off')\n"
        "plt.tight_layout()"
    ),
    md("## 2. Валидационный сплит (по `fire_event_id`, не по пикселям/случайным чипам)"),
    code(
        "from sklearn.model_selection import train_test_split\n\n"
        "events = meta_bs[['chip_id', 'fire_event_id']].copy()\n"
        "unique_events = events['fire_event_id'].dropna().unique()\n"
        "train_events, val_events = train_test_split(unique_events, test_size=0.2, random_state=RANDOM_SEED)\n"
        "train_ids = events.loc[events['fire_event_id'].isin(train_events), 'chip_id']\n"
        "val_ids = events.loc[events['fire_event_id'].isin(val_events), 'chip_id']\n"
        "print(len(train_ids), 'train /', len(val_ids), 'val (уникальных событий:', len(unique_events), ')')"
    ),
    md(
        "## 3. Пиксельный LightGBM — основная модель BS\n\n"
        "**Почему не сеть.** Степень поражения — по сути попиксельная спектральная задача: "
        "dNBR, условленный типом покрова (сама постановка говорит, что фиксированные шкалы "
        "dNBR не работают, потому что лесу/степи/пашне нужна разная калибровка). Сиамский "
        "U-Net видит 179 изображений и голодает по данным, а градиентный бустинг видит "
        "**~47 млн размеченных пикселей** из тех же 179 чипов — и это ровно та часть `Score`, "
        "где был главный провал (`mIoU_sev`).\n\n"
        "Замер на валидации (реальный эксперимент, не предположение):\n\n"
        "| модель | IoU_burn | mIoU_sev | 0.35·IoU+0.30·mIoU |\n"
        "|---|---|---|---|\n"
        "| baseline (единый порог dNBR) | 0.393 | 0.180 | 0.192 |\n"
        "| сиамский U-Net (ансамбль из 2 + TTA) | 0.526 | ~0.30 | ~0.274 |\n"
        "| **пиксельный LightGBM** | **0.546** | **0.556** | **0.358** |\n\n"
        "`mIoU_sev` почти удвоился. Плюс инференс ~0.9 с/чип против ~4.4 с/чип у одной сети "
        "(а с ансамблем и TTA — в разы дороже), что важно для раздела «Скорость» критериев.\n\n"
        "Обучающая таблица — **стратифицированная выборка пикселей**: до "
        "`BS_LGB_MAX_PER_CLASS` пикселей каждого класса с каждого чипа. Это и решает "
        "дисбаланс (класс 0 занимает ~90% площади), и держит таблицу в разумных "
        "размерах по памяти. Из-за сбалансированной выборки вероятности модели "
        "откалиброваны под равномерный prior, а не под реальный — поэтому оптимальный "
        "порог «есть гарь» подбирается отдельно в секции 5 и обычно оказывается высоким."
    ),
    code(
        "BS_LGB_MAX_PER_CLASS = 5000   # пикселей каждого класса с каждого чипа\n"
        "BS_LGB_BOUNDARY_EXTRA = 3000  # доп. пиксели у ГРАНИЦЫ гари (см. ниже)\n"
        "BS_LGB_NUM_LEAVES = 127\n"
        "BS_LGB_ROUNDS = 700\n\n\n"
        "def build_bs_training_table(source, chip_ids, max_per_class=BS_LGB_MAX_PER_CLASS,\n"
        "                             boundary_extra=BS_LGB_BOUNDARY_EXTRA,\n"
        "                             seed=RANDOM_SEED, desc='BS training table'):\n"
        "    \"\"\"Стратифицированная выборка пикселей + дополнительная выборка у границы гари.\n\n"
        "    Случайные пиксели класса 0 почти все далеко от контура, и модель учится на\n"
        "    лёгких примерах. Кольцо снаружи гари и полоса внутри - это как раз те пиксели,\n"
        "    на которых решается IoU_burn (аналог hard negative mining в модуле AF).\n"
        "    Замер на валидации: 0.3660 -> 0.3737 по вкладу модуля в Score.\"\"\"\n"
        "    rng = np.random.default_rng(seed)\n"
        "    X_parts, y_parts = [], []\n"
        "    for chip_id in tqdm(list(chip_ids), desc=desc):\n"
        "        chip = source.load_bs(chip_id)\n"
        "        feats = build_bs_pixel_features(chip)\n"
        "        mask = chip['mask']\n"
        "        valid = nodata_to_ignore(mask, chip['mask_nodata'])\n"
        "        flat = feats.reshape(feats.shape[0], -1)\n"
        "        mask_flat, valid_flat = mask.ravel(), valid.ravel()\n\n"
        "        for c in range(4):\n"
        "            idx = np.flatnonzero((mask_flat == c) & valid_flat)\n"
        "            if len(idx) == 0:\n"
        "                continue\n"
        "            take = min(len(idx), max_per_class)\n"
        "            sel = rng.choice(idx, size=take, replace=False)\n"
        "            X_parts.append(flat[:, sel].T)\n"
        "            y_parts.append(np.full(take, c, dtype=np.int32))\n\n"
        "        burn = mask > 0\n"
        "        if boundary_extra and burn.any():\n"
        "            outer_ring = ndi.binary_dilation(burn, iterations=6) & (~burn)\n"
        "            inner_band = burn & (~ndi.binary_erosion(burn, iterations=6))\n"
        "            for zone in (outer_ring, inner_band):\n"
        "                zone_idx = np.flatnonzero(zone.ravel() & valid_flat)\n"
        "                if len(zone_idx) == 0:\n"
        "                    continue\n"
        "                take = min(len(zone_idx), boundary_extra)\n"
        "                sel = rng.choice(zone_idx, size=take, replace=False)\n"
        "                X_parts.append(flat[:, sel].T)\n"
        "                y_parts.append(mask_flat[sel].astype(np.int32))\n"
        "    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)\n\n\n"
        "# лёгкая проверка на 2 чипах - фичи строятся, размерности сходятся\n"
        "_bs_X_smoke, _bs_y_smoke = build_bs_training_table(src, list(train_ids[:2]), max_per_class=500,\n"
        "                                                     desc='smoke test (2 чипа)')\n"
        "print('smoke test:', _bs_X_smoke.shape, 'классы:', np.bincount(_bs_y_smoke, minlength=4),\n"
        "      'NaN:', int(np.isnan(_bs_X_smoke).sum()))"
    ),
    code(
        "if RUN_HEAVY_TRAINING:\n"
        "    import json\n"
        "    import time\n\n"
        "    import lightgbm as lgb\n\n"
        "    X_bs_train, y_bs_train = build_bs_training_table(src, train_ids, desc='BS таблица (train)')\n"
        "    X_bs_val, y_bs_val = build_bs_training_table(src, val_ids, desc='BS таблица (val)')\n"
        "    print('таблицы:', X_bs_train.shape, X_bs_val.shape,\n"
        "          '| классы train:', np.bincount(y_bs_train, minlength=4))\n\n"
        "    _lc_idx = BS_PIXEL_LANDCOVER_IDX\n"
        "    _dtrain = lgb.Dataset(X_bs_train, label=y_bs_train, feature_name=BS_PIXEL_FEATURE_NAMES,\n"
        "                           categorical_feature=[_lc_idx])\n"
        "    _dval = lgb.Dataset(X_bs_val, label=y_bs_val, feature_name=BS_PIXEL_FEATURE_NAMES,\n"
        "                         categorical_feature=[_lc_idx], reference=_dtrain)\n"
        "    bs_lgb_params = dict(\n"
        "        objective='multiclass', num_class=4, metric='multi_logloss',\n"
        "        num_leaves=BS_LGB_NUM_LEAVES, learning_rate=0.06,\n"
        "        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,\n"
        "        min_data_in_leaf=100, seed=RANDOM_SEED, verbose=-1, num_threads=0,\n"
        "    )\n"
        "    bs_lgb_model = lgb.train(\n"
        "        bs_lgb_params, _dtrain, num_boost_round=BS_LGB_ROUNDS, valid_sets=[_dval],\n"
        "        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],\n"
        "    )\n"
        "    print('best_iteration:', bs_lgb_model.best_iteration)\n"
        "    _imp = sorted(zip(BS_PIXEL_FEATURE_NAMES, bs_lgb_model.feature_importance()),\n"
        "                   key=lambda kv: -kv[1])[:15]\n"
        "    print('топ-15 признаков:', _imp)\n\n"
        "    bs_lgb_path = os.path.join(CHECKPOINT_DIR, 'bs_lightgbm.txt')\n"
        "    bs_lgb_model.save_model(bs_lgb_path)\n"
        "    with open(os.path.join(CHECKPOINT_DIR, 'bs_lightgbm_meta.json'), 'w', encoding='utf-8') as f:\n"
        "        json.dump({\n"
        "            'feature_names': BS_PIXEL_FEATURE_NAMES,\n"
        "            'categorical_feature_idx': [_lc_idx],\n"
        "            'params': bs_lgb_params,\n"
        "            'best_iteration': bs_lgb_model.best_iteration,\n"
        "            'max_per_class': BS_LGB_MAX_PER_CLASS,\n"
        "            'boundary_extra': BS_LGB_BOUNDARY_EXTRA,\n"
        "            'seed': RANDOM_SEED,\n"
        "            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),\n"
        "        }, f, ensure_ascii=False, indent=2)\n"
        "    print('Сохранено:', bs_lgb_path)\n\n"
        "    # проверка воспроизводимости: модель с диска даёт те же предсказания\n"
        "    _reloaded = lgb.Booster(model_file=bs_lgb_path)\n"
        "    _a = bs_lgb_model.predict(X_bs_val[:200], num_iteration=bs_lgb_model.best_iteration)\n"
        "    _b = _reloaded.predict(X_bs_val[:200], num_iteration=bs_lgb_model.best_iteration)\n"
        "    assert np.allclose(_a, _b), 'сохранённая модель не воспроизводит предсказания!'\n"
        "    print('Проверка воспроизводимости пройдена.')\n"
        "else:\n"
        "    print('RUN_HEAVY_TRAINING=False - обучение LightGBM пропущено (Colab/Kaggle).')\n"
        "    bs_lgb_model = None"
    ),
    md(
        "## 4. Сиамский U-Net — опционально (`BS_TRAIN_UNET`)\n\n"
        "**По умолчанию выключен.** На реальном прогоне ансамбль из двух сиамских U-Net дал "
        "`0.35·IoU+0.30·mIoU ≈ 0.274` против `0.358` у пиксельного LightGBM, при этом обучение "
        "занимало ~40 минут GPU против ~4 минут CPU, а инференс был в разы дороже. Код "
        "оставлен полностью рабочим: если хотите проверить, добавляет ли сеть что-то **в "
        "ансамбле** с LightGBM (пространственная связность контура — то, чего попиксельной "
        "модели не хватает), поставьте `BS_TRAIN_UNET = True`. Секция 5 честно сравнит три "
        "стратегии (LightGBM / U-Net / среднее) на валидации и возьмёт лучшую.\n\n"
        "По сравнению с прошлой версией здесь три исправления: `base=16` вместо 32 (в 4 раза "
        "меньше параметров — на 179 чипах сеть переобучалась), `crops_per_chip` (на эпоху "
        "приходится в 6 раз больше шагов оптимизатора — раньше их было всего ~880 за всё "
        "обучение), и **веса классов по корню из обратной частоты** вместо чистой обратной "
        "(раньше класс «не горело» получал вес 0.045 против ~1.8 у горелых — сеть была "
        "вынуждена предсказывать гарь почти везде, отсюда и порог 0.8)."
    ),
    code(
        "BS_TRAIN_UNET = False   # True - обучить сеть и проверить ансамбль с LightGBM\n"
        "BS_UNET_BASE = 16\n"
        "BS_UNET_EPOCHS = 20\n"
        "BS_UNET_CROPS_PER_CHIP = 6\n"
        "BS_N_MODELS = 1\n\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n\n\n"
        "class BSChipDataset(Dataset):\n"
        "    \"\"\"crops_per_chip > 1 - несколько случайных кропов с одного чипа за эпоху:\n"
        "    на 179 чипах иначе получается всего ~22 шага оптимизатора на эпоху.\"\"\"\n\n"
        "    def __init__(self, source, chip_ids, augment=False, crop_size=None, crops_per_chip=1):\n"
        "        self.source = source\n"
        "        self.chip_ids = list(chip_ids)\n"
        "        self.augment = augment\n"
        "        self.crop_size = crop_size\n"
        "        self.crops_per_chip = crops_per_chip if augment else 1\n\n"
        "    def __len__(self):\n"
        "        return len(self.chip_ids) * self.crops_per_chip\n\n"
        "    def __getitem__(self, i):\n"
        "        chip = self.source.load_bs(self.chip_ids[i % len(self.chip_ids)])\n"
        "        pre, post, static = build_bs_feature_stack_siamese(chip)\n"
        "        mask = chip['mask'].astype(np.int64)\n"
        "        valid = nodata_to_ignore(mask, chip['mask_nodata'])\n"
        "        mask = np.where(valid, mask, 0)\n\n"
        "        if self.crop_size:\n"
        "            h, w = mask.shape\n"
        "            cs = self.crop_size\n"
        "            top = np.random.randint(0, h - cs + 1) if self.augment else (h - cs) // 2\n"
        "            left = np.random.randint(0, w - cs + 1) if self.augment else (w - cs) // 2\n"
        "            pre = pre[:, top:top + cs, left:left + cs]\n"
        "            post = post[:, top:top + cs, left:left + cs]\n"
        "            static = static[:, top:top + cs, left:left + cs]\n"
        "            mask = mask[top:top + cs, left:left + cs]\n"
        "            valid = valid[top:top + cs, left:left + cs]\n\n"
        "        if self.augment:\n"
        "            if np.random.rand() < 0.5:\n"
        "                pre, post, static = pre[:, :, ::-1].copy(), post[:, :, ::-1].copy(), static[:, :, ::-1].copy()\n"
        "                mask, valid = mask[:, ::-1].copy(), valid[:, ::-1].copy()\n"
        "            if np.random.rand() < 0.5:\n"
        "                pre, post, static = pre[:, ::-1, :].copy(), post[:, ::-1, :].copy(), static[:, ::-1, :].copy()\n"
        "                mask, valid = mask[::-1, :].copy(), valid[::-1, :].copy()\n"
        "            k = np.random.randint(0, 4)\n"
        "            pre = np.ascontiguousarray(np.rot90(pre, k, axes=(1, 2)))\n"
        "            post = np.ascontiguousarray(np.rot90(post, k, axes=(1, 2)))\n"
        "            static = np.ascontiguousarray(np.rot90(static, k, axes=(1, 2)))\n"
        "            mask = np.ascontiguousarray(np.rot90(mask, k))\n"
        "            valid = np.ascontiguousarray(np.rot90(valid, k))\n\n"
        "        return (\n"
        "            torch.from_numpy(pre.copy()), torch.from_numpy(post.copy()),\n"
        "            torch.from_numpy(static.copy()), torch.from_numpy(mask.copy()),\n"
        "            torch.from_numpy(valid.astype(np.float32).copy()),\n"
        "        )\n\n\n"
        "class DiceCELoss(nn.Module):\n"
        "    \"\"\"CE + squared Dice + ordinal-штраф (классы степени упорядочены 0<1<2<3),\n"
        "    всё - только на valid-пикселях.\"\"\"\n\n"
        "    def __init__(self, class_weights=None, n_classes=4, squared_dice=True, ordinal_weight=0.2):\n"
        "        super().__init__()\n"
        "        self.n_classes = n_classes\n"
        "        self.squared_dice = squared_dice\n"
        "        self.ordinal_weight = ordinal_weight\n"
        "        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')\n\n"
        "    def forward(self, logits, target, valid):\n"
        "        ce = self.ce(logits, target)\n"
        "        ce = (ce * valid).sum() / valid.sum().clamp_min(1.0)\n\n"
        "        probs = F.softmax(logits, dim=1)\n"
        "        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()\n"
        "        valid_e = valid.unsqueeze(1)\n"
        "        inter = (probs * target_onehot * valid_e).sum(dim=(0, 2, 3))\n"
        "        if self.squared_dice:\n"
        "            union = ((probs ** 2) * valid_e).sum(dim=(0, 2, 3)) + ((target_onehot ** 2) * valid_e).sum(dim=(0, 2, 3))\n"
        "        else:\n"
        "            union = (probs * valid_e).sum(dim=(0, 2, 3)) + (target_onehot * valid_e).sum(dim=(0, 2, 3))\n"
        "        dice_loss = 1.0 - ((2 * inter + 1.0) / (union + 1.0)).mean()\n\n"
        "        class_idx = torch.arange(self.n_classes, device=logits.device, dtype=probs.dtype).view(1, self.n_classes, 1, 1)\n"
        "        expected_class = (probs * class_idx).sum(dim=1)\n"
        "        ordinal = ((expected_class - target.float()) ** 2 * valid).sum() / valid.sum().clamp_min(1.0)\n\n"
        "        return ce + dice_loss + self.ordinal_weight * ordinal\n\n\n"
        "def compute_bs_class_weights(meta_bs, chip_hw=(512, 512)):\n"
        "    \"\"\"Корень из обратной частоты, а не чистая обратная: с чистой класс 0 получал\n"
        "    вес ~0.045 против ~1.8 у горелых классов, и сеть предсказывала гарь почти везде.\"\"\"\n"
        "    total_px = len(meta_bs) * chip_hw[0] * chip_hw[1]\n"
        "    sev_total = meta_bs[['sev1_px', 'sev2_px', 'sev3_px']].sum()\n"
        "    counts = np.array([\n"
        "        total_px - sev_total.sum(), sev_total['sev1_px'], sev_total['sev2_px'], sev_total['sev3_px'],\n"
        "    ], dtype=np.float64)\n"
        "    counts = np.clip(counts, 1, None)\n"
        "    inv = 1.0 / np.sqrt(counts)\n"
        "    weights = inv / inv.sum() * len(counts)\n"
        "    return torch.tensor(weights, dtype=torch.float32)\n\n\n"
        "def compute_bs_sample_weights(meta_bs, chip_ids, chip_hw=(512, 512)):\n"
        "    sub = meta_bs.set_index('chip_id').loc[list(chip_ids)]\n"
        "    severe_frac = (sub['sev2_px'].fillna(0) + sub['sev3_px'].fillna(0)) / (chip_hw[0] * chip_hw[1])\n"
        "    return torch.tensor((1.0 + 5.0 * severe_frac).values, dtype=torch.double)\n\n\n"
        "# Быстрая проверка модели и лосса (НЕ обучение, ~1 сек на CPU)\n"
        "_smoke_model = SiameseUNet(pair_channels=len(BS_PRE_CHANNELS), static_channels=len(BS_STATIC_CHANNELS),\n"
        "                            n_classes=4, base=8)\n"
        "_logits = _smoke_model(torch.randn(2, len(BS_PRE_CHANNELS), 64, 64),\n"
        "                        torch.randn(2, len(BS_PRE_CHANNELS), 64, 64),\n"
        "                        torch.randn(2, len(BS_STATIC_CHANNELS), 64, 64))\n"
        "_loss = DiceCELoss(n_classes=4)(_logits, torch.randint(0, 4, (2, 64, 64)), torch.ones(2, 64, 64))\n"
        "_loss.backward()\n"
        "print('smoke test OK, logits shape:', tuple(_logits.shape), ' loss:', float(_loss.detach()))"
    ),
    code(
        "def train_bs_model(source, train_ids, val_ids, meta_bs, model_idx=0, epochs=BS_UNET_EPOCHS,\n"
        "                    crop_size=256, batch_size=8, lr=1e-3, device=DEVICE, seed=RANDOM_SEED):\n"
        "    torch.manual_seed(seed)\n"
        "    train_ds = BSChipDataset(source, train_ids, augment=True, crop_size=crop_size,\n"
        "                              crops_per_chip=BS_UNET_CROPS_PER_CHIP)\n"
        "    val_ds = BSChipDataset(source, val_ids, augment=False, crop_size=None)\n"
        "    sample_weights = compute_bs_sample_weights(meta_bs, train_ids).repeat(BS_UNET_CROPS_PER_CHIP)\n"
        "    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)\n"
        "    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)\n\n"
        "    model = SiameseUNet(pair_channels=len(BS_PRE_CHANNELS), static_channels=len(BS_STATIC_CHANNELS),\n"
        "                         n_classes=4, base=BS_UNET_BASE).to(device)\n"
        "    criterion = DiceCELoss(class_weights=compute_bs_class_weights(meta_bs).to(device), n_classes=4)\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)\n"
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)\n\n"
        "    best_metric, best_state = -1.0, None\n"
        "    epoch_bar = tqdm(range(epochs), desc=f'U-Net {model_idx}: epochs')\n"
        "    for epoch in epoch_bar:\n"
        "        model.train()\n"
        "        running = 0.0\n"
        "        batch_bar = tqdm(train_loader, desc=f'epoch {epoch+1}/{epochs} (train)', leave=False)\n"
        "        for pre, post, static, mask, valid in batch_bar:\n"
        "            pre, post, static = pre.to(device), post.to(device), static.to(device)\n"
        "            mask, valid = mask.to(device), valid.to(device)\n"
        "            optimizer.zero_grad()\n"
        "            loss = criterion(model(pre, post, static), mask, valid)\n"
        "            loss.backward()\n"
        "            optimizer.step()\n"
        "            running += loss.item() * pre.size(0)\n"
        "            batch_bar.set_postfix(loss=f'{loss.item():.4f}')\n"
        "        scheduler.step()\n\n"
        "        model.eval()\n"
        "        acc_binary, acc_sev = ConfusionAccumulator(classes=(1,)), ConfusionAccumulator(classes=(1, 2, 3))\n"
        "        with torch.no_grad():\n"
        "            for idx in tqdm(range(len(val_ds)), desc=f'epoch {epoch+1}/{epochs} (val)', leave=False):\n"
        "                pre, post, static, mask, valid = val_ds[idx]\n"
        "                pred = model(pre.unsqueeze(0).to(device), post.unsqueeze(0).to(device),\n"
        "                             static.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()\n"
        "                mask_np, valid_np = mask.numpy(), valid.numpy().astype(bool)\n"
        "                acc_binary.update(binarize_burn(pred), binarize_burn(mask_np), valid=valid_np)\n"
        "                acc_sev.update(pred, mask_np, valid=valid_np)\n"
        "        val_metric = 0.35 * iou_burn(acc_binary) + 0.30 * miou_severity(acc_sev)\n"
        "        epoch_bar.set_postfix(loss=f'{running/len(train_ds):.4f}',\n"
        "                               IoU_burn=f'{iou_burn(acc_binary):.3f}',\n"
        "                               mIoU_sev=f'{miou_severity(acc_sev):.3f}')\n"
        "        if val_metric > best_metric:\n"
        "            best_metric = val_metric\n"
        "            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}\n"
        "            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'bs_unet_{model_idx}_best.pt'))\n"
        "    model.load_state_dict(best_state)\n"
        "    return model, best_metric\n\n\n"
        "if RUN_HEAVY_TRAINING and BS_TRAIN_UNET:\n"
        "    import json\n"
        "    import time\n\n"
        "    bs_models = []\n"
        "    for _mi in range(BS_N_MODELS):\n"
        "        _m, _vm = train_bs_model(src, train_ids, val_ids, meta_bs, model_idx=_mi, seed=RANDOM_SEED + _mi)\n"
        "        bs_models.append(_m)\n"
        "        print(f'U-Net {_mi}: best val_metric={_vm:.4f}')\n"
        "    with open(os.path.join(CHECKPOINT_DIR, 'bs_unet_meta.json'), 'w', encoding='utf-8') as f:\n"
        "        json.dump({'n_models': BS_N_MODELS, 'base': BS_UNET_BASE,\n"
        "                    'pre_channels': list(BS_PRE_CHANNELS),\n"
        "                    'static_channels': list(BS_STATIC_CHANNELS),\n"
        "                    'epochs': BS_UNET_EPOCHS, 'seed': RANDOM_SEED,\n"
        "                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, ensure_ascii=False, indent=2)\n"
        "else:\n"
        "    print('BS_TRAIN_UNET=False (или RUN_HEAVY_TRAINING=False) - сеть не обучается, '\n"
        "          'работаем на пиксельном LightGBM.')\n"
        "    bs_models = None"
    ),
    md(
        "## 5. Выбор стратегии и калибровка порога «есть гарь»\n\n"
        "Честно сравниваем на валидации то, что реально доступно: **LightGBM**, **U-Net** "
        "(если обучался) и **среднее их вероятностей**. Берём победителя по "
        "`0.35·IoU_burn + 0.30·mIoU_sev` — это ровно вклад модуля BS в `Score`.\n\n"
        "Порог «есть ли вообще гарь» (`P(1)+P(2)+P(3)`) калибруется **отдельно** от того, "
        "какая степень внутри горелой области (там просто `argmax` среди 1/2/3): `IoU_burn` "
        "и `mIoU_sev` в `Score` взвешены отдельно, и единый компромиссный порог на всё сразу "
        "их обе просаживает. Вероятности по валидации считаются один раз и кэшируются."
    ),
    code(
        "def _bs_probs_for_ids(source, chip_ids, lgb_model=None, unet_models=None, use_tta=False):\n"
        "    \"\"\"Считает вероятности (4,H,W) по каждой доступной модели один раз и кэширует\n"
        "    вместе с маской/valid/landcover - дальше перебор порогов идёт по кэшу.\"\"\"\n"
        "    cached = []\n"
        "    for chip_id in tqdm(list(chip_ids), desc='кэшируем вероятности (val)'):\n"
        "        chip = source.load_bs(chip_id)\n"
        "        entry = {\n"
        "            'mask': chip['mask'],\n"
        "            'valid': nodata_to_ignore(chip['mask'], chip['mask_nodata']),\n"
        "            'landcover': chip['aux']['landcover'],\n"
        "        }\n"
        "        if lgb_model is not None:\n"
        "            entry['lgb'] = predict_bs_proba_lgb(chip, lgb_model)\n"
        "        if unet_models:\n"
        "            pre, post, static = build_bs_feature_stack_siamese(chip)\n"
        "            entry['unet'] = predict_bs_proba_tta(\n"
        "                torch.from_numpy(pre), torch.from_numpy(post), torch.from_numpy(static),\n"
        "                unet_models, device=DEVICE, use_tta=use_tta,\n"
        "            )\n"
        "        cached.append(entry)\n"
        "    return cached\n\n\n"
        "def _bs_eval(cached, key_or_blend, threshold):\n"
        "    acc_b, acc_s = ConfusionAccumulator(classes=(1,)), ConfusionAccumulator(classes=(1, 2, 3))\n"
        "    for e in cached:\n"
        "        if key_or_blend == 'blend':\n"
        "            probs = 0.5 * (e['lgb'] + e['unet'])\n"
        "        else:\n"
        "            probs = e[key_or_blend]\n"
        "        pred = apply_bs_thresholds(probs, burn_threshold=threshold)\n"
        "        acc_b.update(binarize_burn(pred), binarize_burn(e['mask']), valid=e['valid'])\n"
        "        acc_s.update(pred, e['mask'], valid=e['valid'])\n"
        "    ib, ms = iou_burn(acc_b), miou_severity(acc_s)\n"
        "    return 0.35 * ib + 0.30 * ms, ib, ms\n\n\n"
        "BS_THRESHOLDS = np.linspace(0.30, 0.95, 14)\n\n"
        "if RUN_HEAVY_TRAINING and (bs_lgb_model is not None or bs_models):\n"
        "    _cached = _bs_probs_for_ids(src, val_ids, lgb_model=bs_lgb_model, unet_models=bs_models)\n\n"
        "    _strategies = []\n"
        "    if bs_lgb_model is not None:\n"
        "        _strategies.append('lgb')\n"
        "    if bs_models:\n"
        "        _strategies.append('unet')\n"
        "    if bs_lgb_model is not None and bs_models:\n"
        "        _strategies.append('blend')\n\n"
        "    _best = None\n"
        "    for _strategy in _strategies:\n"
        "        _strategy_best = None\n"
        "        for _t in tqdm(BS_THRESHOLDS, desc=f'перебор порога ({_strategy})'):\n"
        "            _s, _ib, _ms = _bs_eval(_cached, _strategy, _t)\n"
        "            if _strategy_best is None or _s > _strategy_best[0]:\n"
        "                _strategy_best = (_s, _ib, _ms, float(_t))\n"
        "            if _best is None or _s > _best[0]:\n"
        "                _best = (_s, _strategy, float(_t), _ib, _ms)\n"
        "        print(f'{_strategy:6s}: лучший score={_strategy_best[0]:.4f} '\n"
        "              f'(IoU_burn={_strategy_best[1]:.4f}, mIoU_sev={_strategy_best[2]:.4f}, '\n"
        "              f'порог={_strategy_best[3]:.2f})')\n\n"
        "    bs_score_val, bs_strategy, burn_threshold, bs_iou_burn_val, bs_miou_sev_val = _best\n"
        "    print()\n"
        "    print(f'ВЫБРАНО: стратегия={bs_strategy}, порог={burn_threshold:.3f}')\n"
        "    print(f'  IoU_burn={bs_iou_burn_val:.4f}  mIoU_sev={bs_miou_sev_val:.4f}  '\n"
        "          f'вклад в Score (0.35*IoU+0.30*mIoU)={bs_score_val:.4f}')\n\n"
        "    _bs_strategy_path = os.path.join(CHECKPOINT_DIR, 'bs_strategy.json')\n"
        "    with open(_bs_strategy_path, 'w', encoding='utf-8') as f:\n"
        "        json.dump({'strategy': bs_strategy, 'burn_threshold': burn_threshold,\n"
        "                    'iou_burn_val': bs_iou_burn_val, 'miou_sev_val': bs_miou_sev_val,\n"
        "                    'score_val': bs_score_val}, f, ensure_ascii=False, indent=2)\n"
        "    print('Стратегия и порог сохранены в', _bs_strategy_path)\n"
        "else:\n"
        "    bs_strategy, burn_threshold = 'lgb', 0.8"
    ),
    md(
        "## 6. Постобработка и проверка, помогает ли она\n\n"
        "Удаление мелких изолированных блобов (`postprocess_bs_mask` из встроенного "
        "`bs_pipeline`) проверяется **на валидации**, а не включается вслепую: на "
        "попиксельной модели фильтрация шума обычно помогает `IoU_burn`, но может "
        "подрезать мелкие настоящие гари. Дилатация классов 2/3 по умолчанию выключена "
        "(она задумывалась под сеть, у которой границы занижены)."
    ),
    code(
        "if RUN_HEAVY_TRAINING and (bs_lgb_model is not None or bs_models):\n"
        "    print('без постобработки: score=%.4f' % bs_score_val)\n"
        "    _best_pp = (bs_score_val, 0)\n"
        "    for _mb in [10, 20, 50, 100]:\n"
        "        _acc_b, _acc_s = ConfusionAccumulator(classes=(1,)), ConfusionAccumulator(classes=(1, 2, 3))\n"
        "        for _e in _cached:\n"
        "            _probs = 0.5 * (_e['lgb'] + _e['unet']) if bs_strategy == 'blend' else _e[bs_strategy]\n"
        "            _pred = apply_bs_thresholds(_probs, burn_threshold=burn_threshold)\n"
        "            _pred = postprocess_bs_mask(_pred, min_blob_size=_mb, dilate_classes=())\n"
        "            _acc_b.update(binarize_burn(_pred), binarize_burn(_e['mask']), valid=_e['valid'])\n"
        "            _acc_s.update(_pred, _e['mask'], valid=_e['valid'])\n"
        "        _s = 0.35 * iou_burn(_acc_b) + 0.30 * miou_severity(_acc_s)\n"
        "        print(f'min_blob_size={_mb}: score={_s:.4f} (IoU_burn={iou_burn(_acc_b):.4f}, '\n"
        "              f'mIoU_sev={miou_severity(_acc_s):.4f})')\n"
        "        if _s > _best_pp[0]:\n"
        "            _best_pp = (_s, _mb)\n"
        "    bs_min_blob_size = _best_pp[1]\n"
        "    print()\n"
        "    print('ВЫБРАНО min_blob_size =', bs_min_blob_size, '(0 = постобработка не помогает)')\n"
        "    with open(os.path.join(CHECKPOINT_DIR, 'bs_strategy.json'), encoding='utf-8') as f:\n"
        "        _sj = json.load(f)\n"
        "    _sj['min_blob_size'] = int(bs_min_blob_size)\n"
        "    _sj['score_val_postproc'] = float(_best_pp[0])\n"
        "    with open(os.path.join(CHECKPOINT_DIR, 'bs_strategy.json'), 'w', encoding='utf-8') as f:\n"
        "        json.dump(_sj, f, ensure_ascii=False, indent=2)\n"
        "else:\n"
        "    bs_min_blob_size = 0"
    ),
    md(
        "## 7. Baseline (единый порог dNBR) — контрольная точка"
    ),
    code(
        "def baseline_bs_predict(dnbr, thresholds=(0.1, 0.27, 0.66)):\n"
        "    t0, t1, t2 = thresholds\n"
        "    out = np.zeros(dnbr.shape, dtype=np.uint8)\n"
        "    out[(dnbr >= t0) & (dnbr < t1)] = 1\n"
        "    out[(dnbr >= t1) & (dnbr < t2)] = 2\n"
        "    out[dnbr >= t2] = 3\n"
        "    return out\n\n\n"
        "acc_binary_bl, acc_sev_bl = ConfusionAccumulator(classes=(1,)), ConfusionAccumulator(classes=(1, 2, 3))\n"
        "for chip in tqdm(iter_chips(src, 'bs', val_ids), total=len(val_ids), desc='baseline eval (val)'):\n"
        "    dnbr_c = _nbr(chip['s2_pre']) - _nbr(chip['s2_post'])\n"
        "    pred_c = baseline_bs_predict(dnbr_c)\n"
        "    valid = nodata_to_ignore(chip['mask'], chip['mask_nodata'])\n"
        "    acc_binary_bl.update(binarize_burn(pred_c), binarize_burn(chip['mask']), valid=valid)\n"
        "    acc_sev_bl.update(pred_c, chip['mask'], valid=valid)\n"
        "_bl_score = 0.35 * iou_burn(acc_binary_bl) + 0.30 * miou_severity(acc_sev_bl)\n"
        "print('baseline: IoU_burn=%.4f  mIoU_sev=%.4f  вклад в Score=%.4f' % (\n"
        "    iou_burn(acc_binary_bl), miou_severity(acc_sev_bl), _bl_score))"
    ),
    md(
        "## 8. Быстрый baseline-сабмишн (без обучения, без GPU)\n\n"
        "Единый порог dNBR из шага 7 — можно прогнать по тесту за секунды и получить "
        "контрольную точку на лидерборде, пока основная модель обучается."
    ),
    code(
        "_bl_test_source = resolve_test_source(DATASET_DIR)\n"
        "_bl_sample_submission = _bl_test_source.read_csv('sample_submission.csv')\n"
        "_bl_bs_ids = _bl_sample_submission.loc[\n"
        "    _bl_sample_submission['chip_id'].str.startswith('BS_'), 'chip_id'\n"
        "].unique()\n\n"
        "_baseline_rows = []\n"
        "for chip_id in tqdm(_bl_bs_ids, desc='baseline-инференс BS на тесте'):\n"
        "    chip = _bl_test_source.load_bs(chip_id)\n"
        "    dnbr_c = _nbr(chip['s2_pre']) - _nbr(chip['s2_post'])\n"
        "    pred_c = baseline_bs_predict(dnbr_c)\n"
        "    _baseline_rows.extend(submission_row(chip_id, c, pred_c == c) for c in (1, 2, 3))\n\n"
        "submission_bs_baseline = pd.DataFrame(_baseline_rows)\n"
        "_baseline_path = os.path.join(CHECKPOINT_DIR, 'submission_bs_baseline.csv')\n"
        "submission_bs_baseline.to_csv(_baseline_path, index=False)\n"
        "_bl_bs_sample = _bl_sample_submission[_bl_sample_submission['chip_id'].isin(_bl_bs_ids)]\n"
        "print('Baseline-сабмишн BS сохранён:', _baseline_path, submission_bs_baseline.shape)\n"
        "print('Проблемы валидации (пусто = ок):', validate_submission(submission_bs_baseline, _bl_bs_sample))\n\n"
        "_af_baseline_path = os.path.join(CHECKPOINT_DIR, 'submission_af_baseline.csv')\n"
        "if os.path.exists(_af_baseline_path):\n"
        "    _final_baseline = pd.concat([pd.read_csv(_af_baseline_path), submission_bs_baseline], ignore_index=True)\n"
        "    _final_baseline_path = os.path.join(CHECKPOINT_DIR, 'submission_baseline.csv')\n"
        "    _final_baseline.to_csv(_final_baseline_path, index=False)\n"
        "    print('Полный baseline submission.csv собран:', _final_baseline_path, _final_baseline.shape)\n"
        "    print('Проблемы полной валидации (пусто = ок):',\n"
        "          validate_submission(_final_baseline, _bl_sample_submission))\n"
        "else:\n"
        "    print('submission_af_baseline.csv ещё нет - прогоните секцию baseline-сабмишна ноутбука 1.')\n"
        "_bl_test_source.close()"
    ),
    md(
        "## 9. Инференс на тесте + сборка итогового submission.csv\n\n"
        "Использует выбранную в секции 5 стратегию и откалиброванные порог/постобработку. "
        "Если ничего не обучалось в этой сессии (`RUN_HEAVY_TRAINING=False`) — модель, "
        "стратегия и порог подхватываются из `CHECKPOINT_DIR`, так что инференс можно "
        "гонять отдельно от обучения. В конце, если `submission_af.csv` из ноутбука 1 уже "
        "лежит рядом, обе части собираются в финальный `submission.csv`."
    ),
    code(
        "import json\n\n"
        "if bs_lgb_model is None and bs_models is None:\n"
        "    _lgb_ckpt = os.path.join(CHECKPOINT_DIR, 'bs_lightgbm.txt')\n"
        "    _lgb_meta = os.path.join(CHECKPOINT_DIR, 'bs_lightgbm_meta.json')\n"
        "    if os.path.exists(_lgb_ckpt) and os.path.exists(_lgb_meta):\n"
        "        import lightgbm as lgb\n"
        "        bs_lgb_model = lgb.Booster(model_file=_lgb_ckpt)\n"
        "        with open(_lgb_meta, encoding='utf-8') as f:\n"
        "            bs_lgb_model.best_iteration = json.load(f)['best_iteration']\n"
        "        print('LightGBM подхвачен из чекпоинта:', _lgb_ckpt)\n"
        "    _strategy_path = os.path.join(CHECKPOINT_DIR, 'bs_strategy.json')\n"
        "    if os.path.exists(_strategy_path):\n"
        "        with open(_strategy_path, encoding='utf-8') as f:\n"
        "            _sj = json.load(f)\n"
        "        bs_strategy = _sj.get('strategy', 'lgb')\n"
        "        burn_threshold = _sj.get('burn_threshold', 0.8)\n"
        "        bs_min_blob_size = _sj.get('min_blob_size', 0)\n"
        "        print('Стратегия из чекпоинта:', bs_strategy, '| порог:', burn_threshold,\n"
        "              '| min_blob_size:', bs_min_blob_size)\n\n"
        "if bs_lgb_model is not None or bs_models:\n"
        "    test_source = resolve_test_source(DATASET_DIR)\n"
        "    sample_submission = test_source.read_csv('sample_submission.csv')\n"
        "    bs_test_ids = sample_submission.loc[\n"
        "        sample_submission['chip_id'].str.startswith('BS_'), 'chip_id'\n"
        "    ].unique()\n\n"
        "    bs_rows = []\n"
        "    for chip_id in tqdm(bs_test_ids, desc='инференс BS на тесте'):\n"
        "        chip = test_source.load_bs(chip_id)\n"
        "        if bs_strategy == 'lgb':\n"
        "            probs = predict_bs_proba_lgb(chip, bs_lgb_model)\n"
        "        else:\n"
        "            pre, post, static = build_bs_feature_stack_siamese(chip)\n"
        "            probs_unet = predict_bs_proba_tta(\n"
        "                torch.from_numpy(pre), torch.from_numpy(post), torch.from_numpy(static),\n"
        "                bs_models, device=DEVICE, use_tta=True,\n"
        "            )\n"
        "            probs = probs_unet if bs_strategy == 'unet' else 0.5 * (\n"
        "                predict_bs_proba_lgb(chip, bs_lgb_model) + probs_unet)\n"
        "        pred_mask = apply_bs_thresholds(probs, burn_threshold=burn_threshold)\n"
        "        if bs_min_blob_size:\n"
        "            pred_mask = postprocess_bs_mask(pred_mask, min_blob_size=bs_min_blob_size, dilate_classes=())\n"
        "        bs_rows.extend(submission_row(chip_id, c, pred_mask == c) for c in (1, 2, 3))\n\n"
        "    submission_bs = pd.DataFrame(bs_rows)\n"
        "    bs_sub_path = os.path.join(CHECKPOINT_DIR, 'submission_bs.csv')\n"
        "    submission_bs.to_csv(bs_sub_path, index=False)\n"
        "    bs_sample = sample_submission[sample_submission['chip_id'].isin(bs_test_ids)]\n"
        "    print('Сохранено:', bs_sub_path, submission_bs.shape)\n"
        "    print('Проблемы валидации (пусто = ок):', validate_submission(submission_bs, bs_sample))\n\n"
        "    af_sub_path = os.path.join(CHECKPOINT_DIR, 'submission_af.csv')\n"
        "    if os.path.exists(af_sub_path):\n"
        "        final_submission = pd.concat([pd.read_csv(af_sub_path), submission_bs], ignore_index=True)\n"
        "        final_path = os.path.join(CHECKPOINT_DIR, 'submission.csv')\n"
        "        final_submission.to_csv(final_path, index=False)\n"
        "        print('ФИНАЛЬНЫЙ submission.csv собран:', final_path, final_submission.shape)\n"
        "        print('Проблемы полной валидации (пусто = ок):',\n"
        "              validate_submission(final_submission, sample_submission))\n"
        "    else:\n"
        "        print('submission_af.csv ещё нет в CHECKPOINT_DIR - прогоните ноутбук 1, '\n"
        "              'затем повторно выполните эту ячейку.')\n"
        "    test_source.close()\n"
        "else:\n"
        "    print('Нет ни обученной модели, ни чекпоинта - инференс пропущен. '\n"
        "          'Поставьте RUN_HEAVY_TRAINING=True и прогоните секцию 3.')"
    ),
    code("src.close()"),
]

with open("02_bs_burn_severity.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb(bs_cells), f)

print("notebooks written")
