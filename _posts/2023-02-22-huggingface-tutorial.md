# Huggingface Tutorial

This is a tutorial I wrote for my club, I2. I figured it might be able to help
some people out! Thanks for reading! (You can download the original .ipynb
notebook [here](/assets/hf-tutorial.ipynb).)

\- Carter Swartout, I2

# What is this notebook about?
Hugging Face is a popular library and resouce for training and using AI models. While it has many valuable resources, it can be extremely difficult to use. This notebook aims to serve as an introduction to Hugging Face and all the tools it provides.

# Setup

You're going to need to have a Hugging Face account. If you don't have one already, sign up [here](https://huggingface.co/join)!


```python
# install the necessary libraries!
!pip install transformers
!pip install datasets
```

# Basics of HuggingFace

(IMO) Hugging Face serves two tasks: storage of AI resouces (models, tokenizers, datasets) and a library of tools for training/using AI models. These resources take the form of a Github-like repository service (the HF Hub) in addition to libraries.

Hugging Face's prominent library is [transformers](https://github.com/huggingface/transformers/), a library containing powerful foundation (pretrained) models and tools to use them. Hugging Face also has a [tokenizers](https://github.com/huggingface/tokenizers) library for tokenizers, a [datasets](https://github.com/huggingface/datasets) library for datasets, and a [diffusers](https://github.com/huggingface/diffusers) library for, you guessed it, diffusion models. (They have a lot of stuff. Too much stuff IMO. It is a bit overwhelming.)

Hugging Face has the [Hub](https://huggingface.co/docs/hub/index), a Github-like service for storing models, datasets, and more. You can use this to store trained models or datasets or access others pre-trained models!

The good of Hugging Face:

- Easy to store and manage models and datasets.
- Has many important pre-trained models and popular datasets.
- Has some really powerful tools.

The bad of Hugging Face:
- The documentation can be terrible.
- Incredibly confusing - there's too much going on.
- Poor abstractions - too often I have to *give in to the Great Hugging Face* and just trust that it works.
- Little consistency in classes, data structures, etc.

# Let's use a pretrained model!

Alright, enough talking. Let's get to something fun!

The first way to use a model is with a [`pipeline`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/pipelines). A `pipeline` is a crazy abstraction that reduces a bunch of "scary AI stuff" into a simple object for inference. We simply need to give the `pipeline` a task or model at instantiation and it is ready for inference. Take a look:


```python
from transformers import pipeline

pipe = pipeline("text-classification") # text-classification is the task
pipe(['Wow, this notebook is amazing!', 'I hate self-referential jokes!']) # inference
```

There's a couple things to take note of:

First, we gave it a task, "text-classification". There are many different [tasks](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/pipelines#transformers.pipeline.task) such as text generation, text classification, and visual tasks. When `pipeline` is instantiated with a task it actually creates a specific pipeline for the task - in this case a [`TextClassificationPipeline`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/pipelines#transformers.TextClassificationPipeline).

Pipelines for different tasks require different arguments when being called. Text-classification pipelines require either a single string or a list of strings when being called. Make sure to check the docs for the specific type of `pipeline`.

When we give it a task without specifying a model it defaults to one. For "text-classification" it defaults to "distilbert", a type of BERT model. If we want something other than the default, we can pass a model name at instantiation: `pipe = pipeline(model=model_name)`

Let's look at another example: If I'm speaking with my German friends, I might like to use sentiment classification what they're saying in German. Fortunately, there's a pretrained [model](https://huggingface.co/oliverguhr/german-sentiment-bert) for that!


```python
pipe = pipeline(model='oliverguhr/german-sentiment-bert') # instantiate with model name
pipe('Carter, ich hasse deinen Humor!') # we can run inference with just a string!
```

Yikes, looks bad...

Let's turn to a different task, generating text! If we want to generate from the following prompt:

`AP News: The University of Washington recently announced`

We can create a new type of pipeline!


```python
# create a text generation pipeline!
pipe = pipeline('text-generation')
pipe('AP News: The University of Washington recently announced')
```

# Our first step under the hood: transformers and tokenizers

Good work! Let's dive a bit deeper into what actually happens inside the pipeline!


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# example pipeline using AutoModel and AutoTokenizer
class TextPipe:
    def __init__(self, model):
        # download models and tokenizers
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def __call__(self, prompt):
        # make sure it is a list
        if type(prompt) is str:
            prompt = [prompt]
        
        # generate 
        outputs = []
        for p in prompt:
            # tokenize the prompt
            tokenized_prompt = self.tokenizer(p, return_tensors='pt')
            # forward pass through the model
            gen_tensor = self.model.generate(tokenized_prompt['input_ids'])
            # decode the model outputs
            print(gen_tensor[0])
            gen_text = self.tokenizer.decode(gen_tensor[0])
            outputs.append(gen_text)
            # note that we could pass everything in a batch, but i want to be explicit
        
        return outputs
```


```python
# let's try it out!

pipe = TextPipe(model='gpt2')
pipe(['amazon.com is the', 'AI will eventually'])
```

There are two stored fields: `model` and `tokenizer`. `model` comes from: `AutoModelForCausalLM`, a HF class for loading AI models. In this case it loads a pretrained GPT2. `AutoTokenizer` does something similar, loading a tokenizer for GPT2. These AutoThings basically instantiate a class, loading weights or configurations from for them. There are multiple types of AutoThings, but I'll mainly focus on lanugage generation for the rest of this notebook.

Let's first look at `AutoTokenizer`. This is an object that can encode and decode plaintext into tensors which can be used by models. You need to instanitate it with `AutoTokenizer.from_pretrained(name)`, loading `name`'s associated tokenizer from the HF Hub. Often these will be some form of a GPT2 tokenizer (it is exactly that in this case).

There's two important methods you should know: First, simply calling `tokenizer(input)` encodes a string or list of strings. One must specify the flag `return_tensors='pt'` to return PyTorch tensors **THIS IS IMPORTANT**. The output will be a dict containing keys `input_ids` and `attention_mask`. These keys point to PyTorch tensors which can be passed into a model later.


```python
tokenizer = AutoTokenizer.from_pretrained('gpt2') # load gpt2 tokenizer
out = tokenizer('This is an example text', return_tensors='pt') # one example (string)
out # returns a dict with input_ids and attention_mask pointing to tensors
```

If we pass in multiple strings in a list, we need to make sure they're the same length. If they aren't, we'll get an error:


```python
# they must be the same length
tokenizer(['This is example one', 'I am example two!'], return_tensors='pt')
```

To tokenize texts which are different lengths, we need to tell the model how to deal with that. There's two main options - truncate to a certain length, or pad (with special tokens) to a certain length ([docs](https://huggingface.co/docs/transformers/pad_truncation)). For now, I'll pad to the longest sequence in the batch. To do so, I'll need to pass the argument `padding=True`.


```python
# again, but with padding!
tokenizer(['This is example one', 'I am example two!'], return_tensors='pt', padding=True)
```

Shoot, we need to tell the model what token to pad with. A typical choice is the tokenizer's end of sentence or `eos` token. We can set it like this:


```python
tokenizer.pad_token = tokenizer.eos_token # set pad token to eos token
# we try again!
out = tokenizer(['This is example one', 'I am example two!'], return_tensors='pt', padding=True)
out
```

It works! Note that the returned tensors now have first dimension of two, because we passed in two inputs.


```python
out['input_ids'].shape
```

Now for the second important method: `tokenizer.decode(input)` (and `tokenizer.batch_decode`). We want to be able to decode outputs from the model - this is the method for that!

The `input` for `tokenizer.decode(input)` should be a PyTorch tensor of encoded text with **one** dimension. We can first encode text to get a tensor, then decode it and it should be the same!


```python
# encode text
tokenized_text = tokenizer('This is example text', return_tensors='pt')

# decode text
tokenizer.decode(tokenized_text['input_ids'])
```

Shoot, I forgot that `tokenizer(input)` outputs a batch dimension always, even if it is one. Let's index into the first dimension and try again.


```python
# encode text
tokenized_text = tokenizer('This is example text', return_tensors='pt')

print(tokenized_text['input_ids'].shape)

# decode text
tokenizer.decode(tokenized_text['input_ids'][0]) # index into first dim this time
```

Sweet! What if we want to decode *using* a batch? Well, you guessed it, use: `tokenizer.batch_decode(input)`. This expects a batch dimension - let's give it one!


```python
# encode text
tokenized_text = tokenizer(
    ['I am example one!', 'Do not forget about example two!'],
    return_tensors='pt',
    padding=True
)

print(tokenized_text['input_ids'].shape)

# decode text
tokenizer.batch_decode(tokenized_text['input_ids']) # no need to index
```

Perfect! We can even see the `eos` tokens that were used to pad!

`AutoModelForCausalLM` is pretty similar to `AutoTokenizer`. Again, it loads a type of model from HF Hub that you pass in when you instantiate with `AutoModelForCausalLM.from_pretrained(model_name)`. There's also two methods that I'll highlight here as well!

The first is simply calling `model()`, running a forward pass of the model. It often requires several parameters:
- `input_ids` is a tokenized PyTorch tensor (we saw this from the tokenizer)!
- `attention_mask` is another PyTorch tensor, again created with the tokenizer.
- `labels` is not always required, but allows the `model` to output a loss as well. For language generation, `labels` typically is the same as `input_ids`.

For GPT-2, this will output a data structure containing logits and sometimes a loss.

Let's take a look at this in action!


```python
# load GPT-2 model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# tokenize text
x = tokenizer('What a great input string!', return_tensors='pt')

# forward pass
out = model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])
out
```

Try to ignore the wall of text that just appeared and just focus on the first line. For me it is:

`CausalLMOutputWithCrossAttentions(loss=None, logits=tensor([[[ -37.2172,  -36.8864,  -40.3563,  ...`

The model outputs some sort of weird `CausalLMOutputWithCrossAttentions`. IMO this is a bit confusing, but we roll with it. Let's look inside this data structure.

First, we have `loss=None` No loss was calculated because we didn't pass it `labels` when it was called. We'll see more about that in a moment.

Second, we have the raw `logits`, the next token probabilities for each token in `input_ids`. This is huge because for each input token in `input_ids`, each of the 50,000+ output tokens was given a score.

Let's take a look at how we can get loss in out output. To do so, we need to pass `labels` in as well. As mentioned before, labels will be the same as `input_ids`.


```python
# tokenize text
x = tokenizer('What a great input string!', return_tensors='pt')

# forward pass
out = model(
    input_ids=x['input_ids'],
    attention_mask=x['attention_mask'],
    labels=x['input_ids']
)
out.loss
```

Sweet! If we were training, we could now call `out.loss.backward()` and run backpropagation.

The second method that is important is `model.generate()`. This allows us to generate text using our model. We can call it without any input, allowing it to ramble on its own!


```python
# generate text
model.generate()
```

Right, it outputs a tensor with so we need to decode it using the tokenizer. There's a batch dim, so we should index in!


```python
# generate text
out = model.generate()

# decode output
tokenizer.decode(out[0])
```

Nice! If we want to prompt it, we can encode text then pass the tensor into the model when generating.


```python
# encode prompt
prompt = tokenizer('The UW is the best', return_tensors='pt')

# generate text
out = model.generate(**prompt)

# decode output
tokenizer.decode(out[0])
```

Perfect. As we can see, we've been getting warnings about not setting a `max_length` or `max_new_tokens`. We can control text generation via a variety of flags ([docs](https://huggingface.co/docs/transformers/main_classes/text_generation))! For this example, we can focus on how many new tokens to generate.

To do so, we can use the `max_new_tokens` and `min_new_tokens` flags.


```python
# encode prompt
prompt = tokenizer('The UW is the best', return_tensors='pt')

# generate text with 50 new tokens
out = model.generate(**prompt, min_new_tokens=50, max_new_tokens=50)

# decode output
tokenizer.decode(out[0])
```

Now that we have a grasp on this, let's take a look at what it would take to fine-tune our own models!

# Training a model - loading a dataset


To train a model, we need data to train on. Fortunately HF has a bunch of datasets on their Hub. (I reread this and it sounded like an ad read. Sorry.) To download a dataset, we can use the `load_dataset` function from the `datasets` library. Lets do so for a dataset on [financial news](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic).


```python
from datasets import load_dataset

ds = load_dataset('zeroshot/twitter-financial-news-topic')
ds
```

Let's train our model to generate tweets that are similar to the dataset. We won't need any of the `label`'s, so we can remove them.


```python
rem_ds = ds.remove_columns('label')
rem_ds
```

Now we need to tokenize the dataset. To do so, we can use `ds.map` to run a function over each example in the dataset. I'm forcing every tokent o 


```python
# tokenization function
def token_func(example):
    return tokenizer(example['text'])
    
# run over entire dataset
tokenized_ds = rem_ds.map(token_func)
tokenized_ds
```

We no longer need the `text` column, so we can remove it.


```python
rem_tokenized_ds = tokenized_ds.remove_columns('text')
rem_tokenized_ds
```

Now we batch texts to be a consistant size (don't worry much about this part). This will reduce our texts down to a small amount of examples because each one was quite short


```python
from itertools import chain

# group texts into blocks of block_size
block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

batched_ds = rem_tokenized_ds.map(group_texts, batched=True)
batched_ds
```

We're going to want a loss, so we will copy the `input_ids` column to the `labels` column as well.


```python
batched_ds['train'] = batched_ds['train'].add_column('labels', batched_ds['train']['input_ids'])
batched_ds
```

Next, we can create a standard PyTorch dataloader from these datasets. I'll use HF's `default_data_collator`. We won't do a validation run so we only create `train_dl`.


```python
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dl = DataLoader(
    batched_ds['train'],
    shuffle=True,
    batch_size=2, # small batch size bc i want to ensure it runs
    collate_fn=default_data_collator
)
```

Finally, we can create a standard training loop and train the model for 100 batches!


```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
dl_iter = iter(train_dl)

#for batch in train_dl: # uncomment to run full epoch
for i in range(100):
    batch = next(dl_iter)
    # push all to device
    batch = {k: batch[k].to(device) for k in batch.keys()}
    # forward pass
    out = model(**batch)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
```

Let's see what our model now produces!


```python
# generate text
out = model.generate(min_new_tokens=30, max_new_tokens=30)
tokenizer.decode(out[0])
```

It needs more training, but you can see that it is starting to learn!

To upload the model, we can use `model.push_to_hub()`. We first need to login to HF using the cli command `huggingface-cli login`.


```python
!huggingface-cli login

model.push_to_hub('username/test_model')
```

Thank you for your time! Let me know if you have any feedback!
