# Corpus bundles

`steam_reviews_framework/run.py` unpacks these two archives into `data/corpora/`
(`data/corpora/README.md` lists what each folder holds).

| Archive | Contents | Licence |
|---|---|---|
| `wikipage.zip` | cleaned English Wikipedia articles of 814 games (`wiki_clean/`) and LLM rewrites of them (`wiki_llm/`, `wiki_variants/`); `wiki_clean_manifest.json` names each file's article (`source_title`) and the sections kept | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the authors are listed in each article's history on en.wikipedia.org |
| `storepage.zip` | Steam store-page descriptions of 1,811 games (`sp_raw/`) and five LLM rewrites of them | the descriptions belong to the games' developers and publishers; not covered by the repository's MIT licence, included only to reproduce the paper |

The root README's section "Licence" says the same for the whole repository.
