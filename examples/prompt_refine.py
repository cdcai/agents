import argparse
import os
import re
import math
import statistics
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable

import polars as pl
import textgrad as tg
from dotenv import load_dotenv
from textgrad.autograd import Module, StringBasedFunction
from textgrad.engine.openai import AzureChatOpenAI

import agents

load_dotenv()

MODEL = "edav-api-share-gpt4-api-nofilter"
STARTING_SYSTEM_PROMPT = """
    You are a helpful language agent that takes malformed CSV input and returns the corrected input.
    Rows may have extra empty columns added or have un-quoted content that leads to a parse error.
    Try to remedy these errors and return the input to be parsed via polars.read_csv()
""".strip()
N_EPOCHS = 3
BATCH_SIZE = 1
# === Get creds, set env vars =================================
agents.openai_creds_ad()
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("GPT4_URL")


class CSVLoss(Module):
    """
    A Hacky CSV comparison loss FN
    """

    def __init__(self) -> None:
        super().__init__()
        self.call_fn = StringBasedFunction(
            fn=self._eval_df, function_purpose="Evaluate CSV response to ground truth"
        )

    @staticmethod
    def extract_csv(input_str: str) -> str:
        """
        Extract just the CSV content from a body of text, assuming it's properly denoted within a ```csv block
        """
        csv_re = re.compile(r"```[cC]sv\n(.+?)```", re.DOTALL)
        return csv_re.findall(input_str)[0]

    @classmethod
    def _eval_df(cls, input_val: tg.Variable, truth: tg.Variable) -> str:
        """
        Evaluate if model response x matches the expected label y
        """
        try:
            x_df = pl.read_csv(StringIO(cls.extract_csv(input_val.get_value())))
        except pl.exceptions.ComputeError as e:
            if "found more fields than defined" in str(e):
                out = f"Parsing input CSV via `polars.read_csv(<input_csv>)` failed because there were >=1 rows with more fields than columns in the CSV header"
                return out
        except Exception as e:
            # Early return if this still fails to parse
            out = f"Parsing input CSV via `polars.read_csv(<input_csv>)` failed with following exception: {str(e)}"
            return out

        y_df = pl.read_csv(StringIO(cls.extract_csv(truth.get_value())))

        if x_df.equals(y_df):
            out = "Answer Correct."
        else:
            out = "Answer parses correctly via `polars.read_csv(<input_csv>)`, but doesn't match expected value"

        return out

    def forward(self, **kwargs):
        return self.call_fn(inputs=kwargs)


def run_epoch(x: tg.Variable, y: tg.Variable, model: tg.BlackboxLLM, eval_fn: Callable):
    """Main Epoch logic"""
    answer = model(x)
    loss = eval_fn(answer, y)
    loss.backward()

class CSVloads:
    def __init__(self, root_path: os.PathLike, split: float = 0.5) -> None:
        pattern = re.compile(f"(\d)+.csv")
        # pattern is /d.csv , /d_correct.csv
        # so find highest /d.csv file
        self.root_path = Path(root_path)
        self.n_examples = int(pattern.findall([file_path.name for file_path in self.root_path.iterdir() if pattern.match(file_path.name)][-1])[0])
        self.examples = []
        for i in range(self.n_examples):
            with open(self.root_path / f"{i + 1}.csv", "r", encoding="utf-8") as file:
                incorrect_file = tg.Variable(
                    f"```csv\n{file.read()}```",
                    requires_grad=False,
                    role_description="Input CSV with syntax errors",
                )
            with open(self.root_path / f"{i + 1}_correct.csv", "r", encoding="utf-8") as file:
                correct_file = tg.Variable(
                    f"```csv\n{file.read()}```",
                    requires_grad=False,
                    role_description="Correct answer",
                )
            self.examples.append((incorrect_file, correct_file))
        
        self.train_len = math.floor(split * self.n_examples)

    def train_iter(self):
        for example in self.examples[:self.train_len]:
            yield example

    def validation_iter(self):
        for example in self.examples[self.train_len:]:
            yield example

class Validation:
    results : dict[str, any]
    def __init__(self, batch_size: int = 2) -> None:
        self.results = {key: [] for key in ["prompt", "train_acc", "val_acc"]}
        self.batch_size = batch_size

    def _eval_sample(self, sample, model: Module, loss: Module):
        answer = model(sample[0])
        loss_val = loss(input_val=answer, truth=sample[1])

        if loss_val.value == "Answer Correct.":
            return 1
        else:
            return 0
    def evaluate(self, samples: Iterable, prompt: tg.variable, model: Module, loss: Module):
        val_results = []
        self.results["prompt"] = prompt.value
        for i, sample in enumerate(samples):
            val_results.append(self._eval_sample(sample, model, loss))
        
        self.results["val_acc"].append(statistics.mean(val_results))

        if len(self.results["val_acc"]) > 1 and (self.results["val_acc"][-1] < self.results["val_acc"][-2]):
            # Revert prompt if accuracy doesn't improve
            prompt.set_value(self.results["prompt"][-2])


def main():
    data_path = Path("data", "csv_corrections")

    csv_data = CSVloads(data_path)

    # === Set up LLM Agents =================
    eng = AzureChatOpenAI(MODEL)
    eng_backprop = AzureChatOpenAI(MODEL)
    tg.set_backward_engine(eng_backprop, override=True)
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="System prompt for a somewhat capable language model that specifies behaviors and strategies for CSV structure correction",
    )
    optimizer = tg.TextualGradientDescent(
        engine=eng_backprop, parameters=[system_prompt]
    )
    loss_fn = CSVLoss()

    model = tg.BlackboxLLM(eng, system_prompt=system_prompt)
    validator = Validation(BATCH_SIZE)
    # Run for N epochs (+1 for starting place)
    for i in range(N_EPOCHS + 1):
        print(f"=== Round {i} ===============")
        # Train
        losses = []
        optimizer.zero_grad()
        for i, (question, correct) in enumerate(csv_data.train_iter()):
            answer = model(question)
            loss = loss_fn(input_val=answer, truth=correct)
            losses.append(loss)
            # BUG: Any more than 2 samples and we run out of tokens
            if i > 0 and i % BATCH_SIZE == 0:
                total_loss = tg.sum(losses)
                total_loss.backward()
                optimizer.step()
                losses = []
                optimizer.zero_grad()

        validator.evaluate(csv_data.validation_iter(), system_prompt, model, loss_fn)
        print(f"current prompt: {system_prompt}")
        print(f"========================")
        print(validator.results)


if __name__ == "__main__":
    main()
