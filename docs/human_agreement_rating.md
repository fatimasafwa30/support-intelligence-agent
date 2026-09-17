# Independent human rating (Milestone 19.3)

The fixed 50-example sample is read from `reports/human_agreement_sample_50.json`.
No resampling occurs. The evaluator must enter all five scores manually; there
are no default or suggested ratings. Judge scores and Golden annotations are
never loaded by this tool.

## Commands (from the project root)

Start **or resume**:

```powershell
.venv\Scripts\python.exe scripts\rate_human_agreement.py
```

View progress without opening a rating session or loading models:

```powershell
.venv\Scripts\python.exe scripts\rate_human_agreement.py --progress
```

Help:

```powershell
.venv\Scripts\python.exe scripts\rate_human_agreement.py --help
```

Each remaining example shows its original position (X/50), ID, query, predicted
intent, historical evidence, and generated/verified reply. The sample stores
metadata only, so the tool reconstructs context using the existing frozen local
classifier/retriever and the controller's deterministic offline reply generator,
as used by the judge evaluation. It stops if the predicted intent changes or no
reply is produced. Keep these artifacts and code unchanged throughout rating.
The verified reply is the controller's reply after its existing URL guard.

For each dimension, the tool displays the exact description and score 1–5
guidance from `configs/reply_evaluation.yaml`. Enter a single integer 1–5 for
groundedness, correctness, relevance, helpfulness, and tone. Enter an optional
short note, or press Enter to skip the note. Only then is that example appended
and flushed to `reports/human_agreement_ratings.jsonl`.

Type `q`, `quit`, or `exit` at any prompt, or press Ctrl+C, to stop safely.
Completed examples remain saved. An unfinished example is presented again on
restart. The tool skips all validated completed IDs, rejects duplicate IDs and
invalid existing records, and never repairs or overwrites them silently.

Only one writing session may run at once. A `.jsonl.lock` file prevents concurrent
sessions and is removed on normal exit/Ctrl+C. If the process is forcibly killed,
first confirm no rating session remains before removing the stale lock file.
Do not remove or edit the ratings file to resume. Corrupt/incomplete disk records
require explicit inspection; the tool stops rather than silently losing data.

No rating session is started during implementation. Automated tests use only
synthetic inputs in temporary files; those are not human evaluations.
