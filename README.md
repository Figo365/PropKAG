# PropKAG

PropKAG generates literature reviews from paper titles, abstracts and keywords. It extracts research propositions, groups related propositions, drafts reviews for each group, and integrates them into a final review. One round of feedback retrieval adds papers that address gaps in the group drafts.

Feedback revision passes the current group draft, its identified information gaps, and the qualifying supplementary papers (titles, complete abstracts and keywords) together to the LLM. The supplied context is saved as `feedback_revision_contexts_round_01.json` in the run's audit directory. A failed feedback-generation request stops the run instead of substituting a template paragraph.

## Installation

Requires Python 3.10 or later. To use the supplied Python 3.11 environment:

```bash
conda env create -f environment.yml
conda activate propkag
```

For an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

## Literature retrieval

Before running PropKAG, import `data/dataset.csv` into [Volcengine VikingDB](https://console.volcengine.com/vikingdb) and create a vector index. Configure a literature-search service backed by this index, and set `LITERATURE_SEARCH_API_ENDPOINT` and `LITERATURE_SEARCH_API_KEY` in `.env`. Use the same corpus for initial and feedback retrieval.

## Configuration

```bash
cp .env.example .env
```

Fill in the keys and endpoints for both LLM profiles, literature search and embeddings. The default LLM is `deepseek-v4-flash`; the embedding model is `text-embedding-v4`. Both LLM profiles need credentials, including when they use the same service.

LLM and embedding services must support `/chat/completions` and `/embeddings`. The literature search adapter sends a POST with the query parameters `api_key`, `search_type`, `query` and `limit`. It expects `code: 200` and a `data.results` array containing paper metadata. To use a different search interface, edit `src/retrieval/semantic_retriever.py`.

Experiment parameters are in `config/experiment.json`. The runner applies these settings over matching environment variables. For service credentials and model names, shell variables take precedence over `.env`.

## Usage

Check the local configuration:

```bash
python run_experiment.py --check
```

Run a review:

```bash
python run_experiment.py --query "Your research topic" --topic-id T01
```

Add `--dry-run` to print the command without calling the services.

For a batch, prepare a UTF-8 TSV file with one `topic_id<TAB>query` per line:

```bash
python run_experiment.py --topics-file topics.tsv
python run_experiment.py --topics-file topics.tsv --start-from T4
```

`--start-from` accepts a topic ID or a row number, starting at 1. Batches stop on an error unless `--continue-on-error` is set.

Reviews and audit files are written to `result/<model>/<topic_id>/`. Use a new topic ID to keep an earlier run. The manifest records the retrieval budget, graph settings and source and prompt hashes; the stage logs record model calls and any fallback behavior.

## Parameters

| Parameter | Value |
|---|---|
| Initial papers requested / retained after deduplication | 40 / 30 |
| Graph similarity threshold | 0.60 |
| Graph edges | All pairs of the same proposition type at or above the threshold |
| Community detection | Leiden |
| Feedback rounds | 1 |
| Maximum gaps per group | 3 |
| Candidates per gap query | 5 |
| Maximum attempts per gap query | 3 |
| LLM temperature | 0.0 |
| Embedding dimensions | 1024 |


## Files

```text
src/                    Pipeline implementation
config/experiment.json  Experiment parameters
config/prompts/         Prompt templates
data/dataset.csv        Paper metadata
run_experiment.py       Single-topic and batch runner
```

`data/dataset.csv` contains paper records. The runner reads batch queries from the separate TSV file passed to `--topics-file`.
