"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import os
import json
import random
import time
import http.client
import urllib.request
import urllib.error

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir


# some endpoints (e.g. the hf-mirror.com community mirror) reject urllib's
# default "Python-urllib/..." User-Agent with HTTP 403, so send a browser-like one
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def download_url_with_retries(url, dest_path, max_attempts=5):
    """
    Download a URL to dest_path, retrying with exponential backoff on transient
    network errors (e.g. http.client.IncompleteRead from a dropped connection).

    The data is streamed to a temporary file and atomically renamed into place
    only on success, so interrupted downloads never leave a partial file at
    dest_path and a re-run starts from a clean slate.
    """
    temp_path = dest_path + ".tmp"
    request = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_USER_AGENT})
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(temp_path, dest_path)
            return
        except urllib.error.HTTPError as e:
            # permanent client errors (e.g. 403/404) won't be fixed by retrying;
            # only rate limits and server hiccups are worth another attempt
            if e.code not in (429, 500, 502, 503, 504):
                raise
            print(f"Attempt {attempt}/{max_attempts} failed downloading {url}: {e}")
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed downloading {url}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if attempt < max_attempts:
            wait_time = 2 ** attempt
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into the local cache directory, and
    read them with pyarrow. Under torchrun, only one rank downloads, the others wait.

    The endpoint is controlled by the HF_ENDPOINT env var (default https://huggingface.co).
    Users in regions where huggingface.co is slow or blocked can set
    HF_ENDPOINT=https://hf-mirror.com to pull data through the community mirror.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    # the manifest is written last, so its existence means the download completed
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single rank acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            if not os.path.exists(manifest_path):
                listing_url = f"{endpoint}/api/datasets/{repo_id}/parquet/{subset}/{split}"
                listing_path = os.path.join(shards_dir, "listing.json")
                download_url_with_retries(listing_url, listing_path)
                with open(listing_path, "r") as f:
                    shard_urls = json.load(f)
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    # shard URLs come back pointing at huggingface.co; rewrite them
                    # to the active endpoint so the mirror is used for the data too
                    shard_url = shard_url.replace("https://huggingface.co", endpoint)
                    print(f"Downloading {shard_url} ...")
                    shard_path = os.path.join(shards_dir, filename)
                    download_url_with_retries(shard_url, shard_path)
                    filenames.append(filename)
                with open(manifest_path, "w") as f:
                    json.dump(filenames, f)
    with open(manifest_path, "r") as f:
        filenames = json.load(f)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="all", split="auxiliary_train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="all", split="auxiliary_train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
