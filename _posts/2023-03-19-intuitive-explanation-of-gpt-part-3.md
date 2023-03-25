# An Intuitive Explanation of GPT Models - Part 3

## WORK IN PROGRESS

### Read [Part One](https://cswartout.com/2022/11/25/intutive-explanation-of-gpt.html) and [Part Two](https://cswartout.com/2022/12/25/intutive-explanation-of-gpt-part-2.html)!

Hi! In this post I'll finish up explaining GPT models, covering training and masking. Thanks for reading!

## Training

Let's take a look at how GPT-style models are trained. Supervised learning entails an example (x), and target (y). In the case of image classification, an example might be a photo of an animal and the target would be the label of the photo, say cat or dog.

GPT models are trained on textual data, often pretrained using vast amounts of the internet. Unlike image classification, there doesn't seem to be as natural of an example and target for raw text. Using our intuition from earlier, we can think of the autocomplete analogy from earlier. Given some start of a message, autocomplete will output a predicted next word. If we have a message, say: *"The quick brown fox jumps over the lazy dog"*, we could evaluate differerent autocomplete engines by typing in different-length starts of the sentence and seeing if the actual next word is predicted. For the message we had, here's what I'd expect:

```
The -> quick
The quick -> brown
The quick brown -> fox
...
The quick brown fox jumps over the lazy -> dog
```

To score these autocomplete engines, we'd type the above messages to the left of the arrow and see if autocomplete predicts the word to the right of it! If autocomplete predicts the word to the right, it did well on that example! If it doesn't predict the true next word then it did bad on that example. These `message -> predictions` serve as our examples and targets when we evaluate these autocomplete engines. This concept of breaking large pieces of text into prefixes (context) and evaluating if autocomplete (or GPT) predicts the next word in the text is essential to how GPT models are trained.

![training examples](/assets/GPT/GPT_00067.jpg)

(The past tense of jump is a typo - whoops.)

As we learned before, GPT models act like these autocomplete engines and can actually be evaluated in the same way! 

![model loss function](/assets/GPT/GPT_00063.jpg)

We can score GPT on text by giving the model prefixes of the text and seeing how highly it predicts the *actual* next word after the prefix! Mathematically, this is done by taking the cross-entropy loss of the 