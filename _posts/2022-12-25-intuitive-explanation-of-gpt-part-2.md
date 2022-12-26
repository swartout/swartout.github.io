# An Intuitive Explanation of GPT Models - Part 2

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

Masked self-attention is the mechanism that allows other words to *pay attention* to each other (and themselves), hence the "self-attention." Words seldom exist alone and context determines what a word's meaning is. To know the meaning of a word, one must also look at the words around it.

Here's an example:

![word with no context](/assets/GPT/GPT_00033.jpg)

Here's a phrase with all other words blanked out, besides the word "He". If I asked you to explain what "He" represents, you'd likely look at me dumbfounded. Without gleaning context from other words, its impossible to figure out what "He" means!

![still word with no context](/assets/GPT/GPT_00035.jpg)

I raise this example to demonstrate that we need to do *something* to figure out the meaning of "He". I'll suggest a process that's reasonable.

![process for words](/assets/GPT/GPT_00036.jpg)

First off, we'd need to know what we're even looking for, some sort of **query**. This query could be compared to  words to see if they're correlated.

![make query](/assets/GPT/GPT_00037.jpg)

At this point, I'll have to reveal the secret sentence... (thank you I2 for listening to me!)

Jokes aside, we need a query for the word "He" (note: this is the query *for* "He", not "He" *getting* queried). "He" refers to a "man, boy, or male animal", so that seems like a good query!

Now, I'll hit you with the most difficult part. We don't want to compare the query directly to other words. Instead, we want to compare the query to a word's identifiers or tags. I'll give you an example for the word "Carter":

When comparing "Carter" and the query "man, boy, or male animal", we don't actually find the similarity between the word "Carter" and the query. Instead, we compare the similarity of Carter's identifiers! There are many ways to identify me: human, male, speaking, etc. This identifier is a **key** for "Carter". Words need these **keys** to compare to queries, rather than just the word itself.

We give each word a **key**, an identifier to what it represents. These then allow us to compare the **query** to each word's **key**, resulting in a similarity **score** for each word (with respect to the original word "He").

![values are now added](/assets/GPT/GPT_00038.jpg)

