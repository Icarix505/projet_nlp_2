# NLP Project on Insurance Customer Reviews

This repository is an NLP project built around **French insurance customer reviews**.

The main idea is pretty simple, even if the pipeline itself becomes a bit bigger after:  
we start from a zip of Excel files containing reviews from different insurers and products, and the goal is to turn that raw material into something usable for analysis and prediction.

So, in practice, the project tries to answer questions like:
- what kind of reviews are in the data,
- what topics come back often,
- what is the sentiment of a review,
- what star rating could be predicted,
- what main subject a review is talking about,
- and how all of this can be explored in a small app.

It was built in a way where the repository is not only “train a model and stop there”.  
It also tries to be navigable. Meaning:
- the data gets rebuilt and cleaned,
- analysis outputs are saved,
- models are exported,
- predictions are saved,
- and then the app and notebooks can read those files without having to recompute everything every time.

So if someone downloads the repo for the first time, the purpose is that they can understand:
- what the project is about,
- how the data moves through the pipeline,
- what gets stored,
- where it gets stored,
- why it was structured like this,
- and how to run both the pipeline and the app.

---

## 1. Project goal

The project is about **customer reviews in the insurance domain**.

The reviews come from several Excel files grouped inside one zip archive.  
Each row is basically a review written by a customer, with some structured fields around it:
- insurer name,
- product name,
- rating,
- publication date,
- original French review,
- English translation,
- and sometimes a manually corrected text.

The project then uses this dataset for three main prediction tasks:

- **Star prediction**  
  Predict a rating from 1 to 5.

- **Sentiment prediction**  
  Predict whether a review is `negative`, `neutral`, or `positive`.

- **Subject assignment**  
  Assign one main topic label like:
  - `Pricing`
  - `Coverage`
  - `Enrollment`
  - `Customer Service`
  - `Claims Processing`
  - `Cancellation`
  - `Other`

And around those tasks, there is also:
- dataset cleaning,
- EDA,
- n-gram analysis,
- topic modeling,
- test prediction export,
- and a Streamlit app for browsing and quick testing.

---

## 2. What the dataset is, concretely

This is not a generic public benchmark dataset in the usual sense.  
It is a collection of **insurance review files** packed into a zip, with multiple Excel sheets/files merged later by the pipeline.

The data is centered on reviews left by customers about insurance companies and products, for example:
- auto insurance,
- health insurance,
- home insurance,
- life insurance,
- and a few smaller product families too.

From the executed notebook outputs currently visible in the project, the processed dataset contains:

- **34,434 rows**
- **24,104 train rows**
- **10,330 test rows**
- **56 insurers**
- **13 products**
- **435 rows with human correction**

A useful detail is that the source data already contains a split in the `type` column:
- `train` = labeled rows
- `test` = unlabeled rows, generally without `note`

So the repository does not create a synthetic train/test split on its own.  
It mostly respects the split already present in the raw files.

That matters because some choices in the code come from this reality.  
For example, the pipeline has to handle both:
- rows where the rating exists,
- and rows where the final prediction has to be produced and exported.

---

## 3. Main columns in the data

### Raw columns
The original files contain fields like:
- `note`
- `auteur`
- `avis`
- `assureur`
- `produit`
- `type`
- `date_publication`
- `date_exp`
- `avis_en`
- `avis_cor`
- `avis_cor_en`

### Processed columns
After preprocessing, the pipeline creates richer columns such as:
- `review_best_fr`
- `review_best_en`
- `review_clean_fr`
- `review_clean_en`
- `review_clean_model`
- `has_human_correction`
- `text_len_chars`
- `text_len_words`
- `year_publication`
- `month_publication`
- `sentiment_label`
- `subject_sbert`
- `subject_sbert_score`
- `subject_rule`
- `subject_keywords`

The cleaned dataset is exported as:

- `data/reviews_processed.csv`

So if someone wants the single most important “final table” of the project, it is probably this file.

---

## 4. Why the preprocessing is done like that

The preprocessing is shaped by the actual data, not by a perfect theoretical setup.

### Main text source
The main modeling language is **French**, because:
- the reviews are originally written in French,
- and the project is supposed to keep the closest version of customer wording.

The logic is:
- use corrected French text (`avis_cor`) when it exists,
- otherwise use original French text (`avis`).

This was chosen because the notebook outputs show there are only **435 corrected rows** out of more than **34k rows**.  
So building the full project around corrected text only would be kind of wasting too much data.

### Why English is kept
The English columns are still useful for:
- manual inspection,
- comparison,
- display in the app,
- and browsing in notebooks.

So they are not the main modeling source, but they are not useless either.

### Metadata features
The pipeline also keeps some simple metadata features, because the raw data clearly gives them:
- insurer
- product
- review length
- publication year/month
- correction flag

This is especially useful for star prediction, because that task is not purely about isolated text meaning.  
Some context around the review does help.

---

## 5. EDA: what it is doing here

The EDA part is not there just to decorate the notebook.  
It is there because the dataset is not balanced and not uniform, so understanding its shape actually matters for the modeling choices.

The current reports and notebook outputs include:
- train / test split
- distribution of star ratings
- distribution of sentiment labels
- review length histogram
- top insurers
- top products
- average rating by insurer
- average rating by product
- top unigrams
- top bigrams
- NMF topics

### Main takeaways from the current outputs

#### 1. The star distribution is uneven
There are many low-star reviews, especially 1-star reviews.  
That makes the exact rating task harder and also explains why the model often predicts a nearby score instead of the exact one.

#### 2. Product distribution is also uneven
`auto` is by far the biggest product family.  
So the dataset is not a balanced representation of all insurance products.

That justifies:
- keeping `produit` as metadata,
- and being careful when reading global metrics, because the dominant product influences the whole picture.

#### 3. Subject distribution is very unbalanced
The final subject counts in `reviews_processed.csv` show:
- `Customer Service` is the biggest class,
- then `Pricing`,
- then `Claims Processing`,
- while `Enrollment` and `Other` are much smaller.

So if some subject outputs look more stable on the large classes, that is not very surprising.

#### 4. Review length varies a lot
The histogram on `text_len_words` shows short and long reviews coexist.  
That supports:
- cleaning and normalization,
- keeping length features,
- and using representations that work on both short and longer text.

---

## 6. Pipeline design: how the project is structured

The repository was structured with one main idea:  
**the pipeline should produce reusable stored outputs, so the app and the notebooks can read them later without retraining everything.**

This is probably the most important design choice in the repo.

Instead of doing everything inside one notebook and having to rerun all cells all the time, the project separates things like this:

- `pipeline.py` builds and saves outputs
- `reports/` stores human-readable analysis outputs
- `artifacts/` stores model-related and app-related outputs
- `data/` stores the cleaned dataset
- notebooks read those saved files
- the Streamlit app also reads those saved files

This makes the project much easier to navigate, because each folder has a role.

---

## 7. What the pipeline stores, where, and why

This section is maybe the most useful one for a new user, because it explains what is produced and why the repo is laid out like this.

## 7.1 `data/`
This folder stores the processed dataset.

Main file:
- `data/reviews_processed.csv`

### Why it is stored here
Because this is the reference cleaned dataset that:
- notebooks can inspect,
- the app can partially reuse,
- and future experiments can start from.

So `data/` is where the pipeline stores the “final table” version of the dataset.

---

## 7.2 `reports/`
This folder stores **analysis outputs meant to be readable**.

Typical contents:
- dataset summary
- top unigrams
- top bigrams
- NMF topics
- average rating tables
- markdown notes on preprocessing choices

### Why it is stored separately
Because these outputs are not model objects.  
They are closer to descriptive analysis and reporting material.

So keeping them in `reports/` makes the repository easier to read:
- if you want graphs, counts, tables, and interpretation support, go there,
- if you want models and serialized objects, go to `artifacts/` instead.

This separation is really on purpose.

---

## 7.3 `artifacts/`
This folder stores **saved objects and prediction-related outputs**.

Typical examples:
- trained models
- vectorizers
- metadata encoders
- saved search artifacts
- classification reports
- confusion matrices
- test predictions
- error examples
- model-comparison tables

### Why it is stored here
Because the app needs these objects to work without rerunning training.  
The notebooks also use them to comment results.

So `artifacts/` is the place for things that are:
- model-linked,
- reusable by code,
- and often serialized.

You can think of it a bit like the “machine-facing output” folder, while `reports/` is more the “human-facing output” folder.

---

## 7.4 Why this storage split is useful
This split between `data/`, `reports/`, and `artifacts/` was done because otherwise the project becomes messy very quickly.

Without that separation, you would end up mixing:
- final tables,
- analysis csv files,
- trained models,
- plots,
- and app dependencies,

all in one place, which becomes painful to maintain.

So even if it looks a little more verbose at first, it helps a lot for:
- debugging,
- browsing the repo,
- understanding dependencies,
- and knowing what can be regenerated.

---

## 8. Modeling choices

## 8.1 Star prediction

The current star model is **not a pure classifier**.

It is a **hybrid regression pipeline** using:
- Sentence-BERT embeddings
- optional Word2Vec embeddings
- TF-IDF + TruncatedSVD features
- metadata features
- `XGBRegressor`

Then the output is rounded and clipped back to 1–5 stars.

### Why regression was chosen
Because stars are ordered values.  
Predicting 4 instead of 5 is still closer than predicting 1 instead of 5, so regression makes sense here.

### Metrics exported
Because of that design, the project exports two types of metrics.

#### Regression metrics
- `r2_train`
- `r2_test`
- `rmse_test`

#### Rounded star metrics
- `accuracy`
- `weighted_f1`
- `macro_f1`

### Current visible results in the notebook
- `r2_train = 0.885333`
- `r2_test = 0.704871`
- `rmse_test = 0.831961`
- `accuracy = 0.454772`
- `weighted_f1 = 0.453681`
- `macro_f1 = 0.429207`

### Important note
The current executed notebook also shows:
- `uses_word2vec = False`
- `word2vec_dim = 0`

So in the final visible run, Word2Vec is disabled, even if the model name still contains `w2v`.  
That is mostly a naming leftover.

### Interpretation
The confusion matrix suggests the model often predicts **neighboring stars**, not absurdly distant ones.  
So the continuous regression performance is stronger than the rounded exact-match performance, which is actually coherent.

---

## 8.2 Sentiment prediction

The chosen sentiment model is:
- **TF-IDF + SGDClassifier**
- saved as `sgd_log`

### Why this model
Because it is:
- fast,
- easy to train,
- easy to explain,
- and already good enough for the dataset.

### Current visible results
- `accuracy = 0.801867`
- `weighted_f1 = 0.744462`
- `macro_f1 = 0.582784`

### Interpretation
The sentiment model handles:
- `negative`
- `positive`

pretty well, but it struggles a lot on:
- `neutral`

The notebook shows very low recall on `neutral`, which means that neutral language tends to get absorbed into one of the two polar classes.

So the model is useful, but you should not read the global accuracy alone and assume all three classes behave equally well.

---

## 8.3 Subject assignment

The current notebook loads subject predictions in this order:
1. `subject_predictions_sbert.csv`
2. fallback to `subject_predictions_zero_shot.csv` only if needed

So the default subject artifact is **SBERT-based**.

Labels are:
- `Pricing`
- `Coverage`
- `Enrollment`
- `Customer Service`
- `Claims Processing`
- `Cancellation`
- `Other`

A detail that can confuse people at first:
the processed dataset still contains several subject-related columns at the same time, including:
- `subject_sbert`
- `subject_sbert_score`
- `subject_rule`
- `subject_keywords`

That is normal in this repo.  
It means the pipeline keeps intermediate and final subject-related information, not only one single field.

---

## 9. Repository structure

## `src/`
Contains the source code.

Most important file:
- `src/pipeline.py`  
  Main end-to-end workflow:
  - unzip data
  - merge files
  - preprocess text
  - build processed dataset
  - run EDA
  - build features
  - train models
  - export outputs

If you want to understand the project logic, this is the first file to open.

---

## `app.py`
Streamlit app for project browsing and testing.

It loads saved files from:
- `artifacts/`
- `reports/`

and gives a UI to:
- inspect the summary,
- test predictions,
- browse similar reviews,
- and search metadata.

---

## `data/`
Stores the processed dataset.

Main file:
- `data/reviews_processed.csv`

---

## `reports/`
Stores readable analysis outputs:
- summaries,
- topic tables,
- n-grams,
- averages,
- markdown notes,
- and other EDA-style exports.

---

## `artifacts/`
Stores code-facing outputs:
- models
- vectorizers
- reports
- confusion matrices
- prediction csv files
- saved metadata/search objects

---

## `notebooks/`
Contains review and exploration notebooks.

Typical use:
- walkthrough notebook
- qualitative notebook with commentary and interpretation

---

## 10. How to run the project

## 10.1 Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 10.2 Install dependencies

If the project provides a `requirements.txt`:

```bash
pip install -r requirements.txt
```

If not, install manually from the imports used in the code, for example:
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- xgboost
- sentence-transformers
- transformers
- torch
- openpyxl
- streamlit

---

## 10.3 Run the pipeline

Example:

```bash
python -m src.pipeline \
  --zip-path "/path/to/Traduction avis clients.zip" \
  --project-root "."
```

If you want to skip Word2Vec:

```bash
python -m src.pipeline \
  --zip-path "/path/to/Traduction avis clients.zip" \
  --project-root "." \
  --skip-word2vec
```

### What this generates
Running the pipeline should produce:
- cleaned data in `data/`
- descriptive outputs in `reports/`
- models and saved artifacts in `artifacts/`

So if someone runs the project from scratch, those are the folders that get populated or refreshed.

---

## 10.4 Launch the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## 11. How to navigate the app

The app is built for quick navigation of the project outputs.

### Overview page
Use it to:
- understand the dataset size,
- inspect high-level distribution information,
- see some EDA summaries,
- and get a quick general picture.

### Predict & Explain page
Use it to:
- paste a review,
- optionally specify insurer and product,
- get predicted stars,
- get sentiment prediction,
- inspect subject prediction,
- and view explanation / similar-review outputs.

### Search page
Use it to:
- browse stored review metadata,
- filter examples,
- inspect existing reviews in the indexed data.

So if you just want to “play with the project a bit”, the app is usually the fastest entrance.

---

## 12. What a new user should read first

A simple reading order that works pretty well is:

1. this README  
2. `src/pipeline.py`  
3. `app.py`  
4. `data/reviews_processed.csv`  
5. `reports/`  
6. `artifacts/`  
7. the qualitative notebook  
8. the app itself

This order gives a decent mental map without having to guess too much.

---

## 13. What can be changed safely, and what is more fragile

### Usually safe to change
- notebook markdown
- README text
- plot formatting
- explanation wording in the app
- some hyperparameters
- TF-IDF settings
- whether to skip Word2Vec
- presentation notebooks

### Change more carefully
- export file names
- artifact structure expected by the app
- metadata-feature shape used by the star model
- subject-column naming consistency
- search artifact structure

Those are the parts most likely to break the alignment between:
- pipeline,
- notebooks,
- and app.

---

## 14. Short summary

This repository is an NLP project on **French insurance customer reviews**.

It takes raw review files and turns them into:
- a cleaned dataset,
- descriptive analysis outputs,
- trained models,
- saved artifacts,
- and an app for exploration.

The key thing to remember is maybe this:
- `pipeline.py` builds everything,
- `data/` stores the cleaned dataset,
- `reports/` stores human-readable analysis outputs,
- `artifacts/` stores model and app dependencies,
- notebooks explain the outputs,
- and `app.py` lets you browse the final project more easily.