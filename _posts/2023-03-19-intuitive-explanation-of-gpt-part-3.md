# An Intuitive Explanation of GPT Models - Part 3

## WORK IN PROGRESS

### Read [Part One](https://cswartout.com/2022/11/25/intutive-explanation-of-gpt.html) and [Part Two](https://cswartout.com/2022/12/25/intutive-explanation-of-gpt-part-2.html)!

Hi! In this post I'll finish up explaining GPT models, explaining training and masking. Thanks for reading!

## Training

Remember the early analogy to a phone's autocomplete, how the model gets a string of text and is predicting the next word? Using this analogy, a "good" model would have meaningful autocomplete predictions and a "bad" model would have poor ones. This is quite similiar to how the loss function for GPT models works in practice!

![gpt loss function](/assets/GPT/GPT_00063.jpg)

Let's consider the sentence "The quick brown fox"