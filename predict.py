import time
from typing import Optional
import subprocess

import torch
import os

from transformers import AutoTokenizer, AutoConfig
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict
from cog import BasePredictor, ConcatenateIterator, Input, Path

from bragi.metric_generator import MetricGenerator
import torch 

# from config import DEFAULT_MODEL_NAME, DEFAULT_CONFIG_PATH, load_tokenizer, load_tensorizer
from subclass import YieldingLlama
from transformers import LlamaForCausalLM

TENSORIZER_WEIGHTS_PATH = "gs://replicate-weights/vicuna-13b/tensorized/vicuna-13b-16fp.tensors"
# TENSORIZER_WEIGHTS_PATH = "models/vicuna-13b-16fp.tensors"  # path from which we pull weights when there's no COG_WEIGHTS environment variable

DEFAULT_CONFIG_PATH = "models/config.json"
TOKENIZER_PATH = "models/"
MODEL_CLS = LlamaForCausalLM

PROMPT_TEMPLATE = "### Human\n{input}\n### Assistant\n"

def maybe_download(path):
    if path.startswith("gs://"):
        output_path = "/tmp/weights.tensors"
        subprocess.check_call(["gcloud", "storage", "cp", path, output_path])
        return output_path
    return path


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if weights is not None and weights.name == "weights":
            # bugfix
            weights = None
        if weights is None and TENSORIZER_WEIGHTS_PATH:
            self.model = self.load_tensorizer(
                weights=maybe_download(TENSORIZER_WEIGHTS_PATH), plaid_mode=True, cls=MODEL_CLS, config_path=DEFAULT_CONFIG_PATH,
            )
        
        elif hasattr(weights, "filename") and "tensors" in weights.filename:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=MODEL_CLS, config_path=DEFAULT_CONFIG_PATH,
            )
        elif hasattr(weights, "suffix") and "tensors" in weights.suffix:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=MODEL_CLS
            )
        elif "tensors" in weights:
            self.model = self.load_tensorizer(
                weights=weights, plaid_mode=True, cls=MODEL_CLS
            )
        else:
            self.model = self.load_huggingface_model(weights=weights)


        self.tokenizer = self.load_tokenizer(TOKENIZER_PATH)
        self.generator = MetricGenerator(model=self.model, tokenizer=self.tokenizer, device=self.device)

    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)

        return tokenizer

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f"loading weights from {weights} w/o tensorizer")
        model = YieldingLlama.from_pretrained(
            weights, cache_dir="pretrained_weights", torch_dtype=torch.float16
        )
        model.to(self.device)
        print(f"weights loaded in {time.time() - st}")
        return model
    
    def load_tensorizer(self, weights, plaid_mode, cls, config_path):
        st = time.time()
        print(f"deserializing weights from {weights}")
        config = AutoConfig.from_pretrained(config_path)

        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )

        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to the model."),
        init_text: str = Input(
            description="Text to initialize the metric constraints with. This determines the number of syllables in each line.",
            default="",
        ),
        syllable_pattern: str = Input(
            description="Pattern of syllables in each line. `0` represents a new line, `1` represents a line with 1 syllable, `2` represents a line with 2 syllables, etc. Must be space delimited, like '5 5 5 0 5 5 5",
            default="",
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=200,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=.9,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens. Overrides top_p if set. Set to `0` to disable.",
            ge=0,
            default=0,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1,
        ),
        no_repeat_ngram_size: int = Input(
            description="If set, this will prevent the model from repeating any ngrams of this size. Set to `0` to disable.",
            ge=0,
            default=2,
        ),
        free_tokens: str = Input(
            description="Space delimited tokens that should be ignored when counting syllables.",
            default="|| ? . , !",
        ),
        seed: int = Input(
            description="Seed for random number generation, set to `0` for random behavior.",
            ge=0,
            default=0,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
    ) -> str:
        
        if seed != 0:
            torch.manual_seed(seed)
            print(f"seed set to {seed}")
        elif seed == 0:
            seed = torch.seed()
            print(f"Randomly set seed to {seed}")
        
        prompt = PROMPT_TEMPLATE.format(input=prompt)

        free_tokens = free_tokens.split(" ")
        if syllable_pattern != "":
            try:
                syllable_pattern = torch.Tensor([float(x) for x in syllable_pattern.split(" ")])
            except ValueError:
                raise ValueError("Syllable pattern must be space delimited integers")
        else:
            syllable_pattern = None
        
        if init_text == "":
            init_text = None

        top_k = None if top_k == 0 else top_k
        no_repeat_ngram_size = None if no_repeat_ngram_size == 0 else no_repeat_ngram_size

        output = self.generator(
            prompt = prompt,
            text_init = init_text,
            syllable_budget = syllable_pattern,
            free_tokens=free_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=no_repeat_ngram_size,
            remove_invalid_values=True,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            max_length = max_length,
            new_line_token='||',
            bad_words_ids=[[8876]],
            repetition_penalty=repetition_penalty,
        )

        if debug:
            if init_text is not None:
                print(f"Syllables per line in `init_text`: {self.generator.calculate_syllable_budget(init_text)}")
            else:
                print(f"Syllables per line in `syllable_pattern`: {syllable_pattern}")
            print(f"Syllables per line in output: {self.generator.calculate_syllable_budget(output)}")
            print(f"Templated prompt:\n{prompt}")
            print(f"cur memory: {torch.cuda.memory_allocated()}")
            print(f"max allocated: {torch.cuda.max_memory_allocated()}")
            print(f"peak memory: {torch.cuda.max_memory_reserved()}")

        if init_text is not None:
            output = self.add_line_breaks(init_text, output)

        return output
    

    def add_line_breaks(self, init_text, output):
        # Split init_text into paragraphs (chunks separated by double line breaks)
        paragraphs = init_text.split('\n\n')

        # Find number of lines in each paragraph
        num_lines_in_paragraphs = [len(p.split('\n')) for p in paragraphs]

        # Split output into lines
        output_lines = output.split('\n')

        # Initialize empty list for new output lines
        new_output_lines = []

        # Initialize line counter
        line_counter = 0

        # Go through each number in num_lines_in_paragraphs
        for num_lines in num_lines_in_paragraphs:
            # Add lines to new_output_lines
            new_output_lines.extend(output_lines[line_counter:line_counter+num_lines])

            # Increment line counter
            line_counter += num_lines

            # If we are not at the end of output_lines, add a line break
            if line_counter < len(output_lines):
                new_output_lines.append('')

        # Join new_output_lines into a string with line breaks
        new_output = '\n'.join(new_output_lines)

        return new_output
    