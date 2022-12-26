# An Intuitive Explanation of GPT Models - Part 1

Last time we left off at what GPT generally is and what type of information *enters* the model.

![outside view](/assets/GPT/GPT_00028.jpg)

In this post we'll go over what happens *inside* it, detailing the seemingly ever-present attenton mechanism.

Here we can view the token embeddings (we know what these are!) entering the model and output probabilities exiting it. Let's take a peek inside the box.

![outside view](/assets/GPT/GPT_00029.jpg)

The internals of a GPT model are relativley simple - repeated "transformer decoder" blocks, finally followed by a linear layer. The linear layer is self-explanatory and we can look at it later. For now, let's examine one of these mystical "transfomer decoder" blocks.

![inside transfomer decoder](/assets/GPT/GPT_00031.jpg)

The block is comprised of two parts: masked self-attention, and a linear layer (feed forward neural network). Each block has the same structure (or architecture), but will contain different valued weights. For now, let's only consider the "masked self-attention" in a conceptual manner. We'll bring it back to implementation in a moment.

### Masked Self-Attention

![masked self-attention](/assets/GPT/GPT_00032.jpg)

