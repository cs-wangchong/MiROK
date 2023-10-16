# MiROK

---

## Step 1: Package Installation
Python Version Required: >= 3.8

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Step 2: Abs-RAR Mining (Optional)
**Note: You have the option to reuse the existing Abs-RAR pairs in `resources/valid_abs_rars.csv` and skip this step.**

To perform Abs-RAR mining, follow these steps:
- Prepare the mining corpus:
    - Option 1: Replace `source_codes = []` in `script/build_corpus.py` with your own source code data, and then run `python script/build_corpus.py`.
    - Option 2: Download our released corpus data from [Google Drive](https://drive.google.com/file/d/1gfxwfsfuVnJo7g6VzJebVKFENq_HjQR6/view?usp=drive_link).

- Run `python script/mine_absrars.py`.

## Step 3: RAR Finding
- (Optional) Modify the set of Abs-RAR pairs in `resources/valid_abs_rars.csv` based on your specific requirements.
- Prepare libraries: Replace the `lib_list` in `script/find_rars.py` with your own library list.
- Run `python script/find_rars.py`.
