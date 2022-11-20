# An Intuitive Explanation of GPT Models

## IN PROGRESS

![Introduction Slide](/_assets/GPT/GPT_00001.jpg)

Hi all! In this post I'll be covering one of the most influential NLP architectures currently used, GPT models! This is an essay version of a talk I gave to Interactive Intelligence at the UW - watch the video [here](https://www.youtube.com/watch?v=SBhetnU1O_I).

---

![high/low level explanation](/_assets/GPT/GPT_00003.jpg)

When I was learning how Transformers and GPT worked, I found it helpful to develop an intuitive feel for the way the model operates, then dive deeper into the specific implementation. I'll be following that same approach here.

While these human representations do not perfectly capture the reasons and workings of GPT, these heuristics aid inital understanding. I believe it is a worthwhile trade to make. As one grows more confident in their understanding, they can discard the human abstractions as they please.

![what is GPT slide](/_assets/GPT/GPT_00005.jpg)

GPT takes in a sequence of words and outputs a prediction for the next token. If the prompt is: "**The quick brown fox jumps over the lazy**", the output would be: "**dog**" (assuming it has been trained correctly).

It is essentially a super-version of the autocomplete on your phone:

![autocomplete 1](/_assets/GPT/GPT_00006.jpg)

Your phone's autocomplete will take in an input: the text in the message box, and produce an output: the possible next-word predictions.

![autocomplete 2](/_assets/GPT/GPT_00007.jpg)

![autocomplete 3](/_assets/GPT/GPT_00008.jpg)

![autocomplete 4](/_assets/GPT/GPT_00009.jpg)

Each time we press the button for a prediction, the phone has viewed the previous words (prompt) and has predicted the next one (output). The next time the button is pressed, we **include** the previous output as part of the new prompt. This is because the model is autoregressive: *it predicts future values based on it's own past outputs*. It does **not** predict the entire message at once, rather one word at a time, building upon itself. GPT works exactly the same way, predicting the next word, then feeding it back into itself until it has reached a max word limit or stopping character.

![autoregressive](/_assets/GPT/GPT_00010.jpg)

Now if you're like me, you might have tried to write a story or message *entirely* using autocomplete. If you repeatedly press autocomplete, it will continue to generate words. Here's an example:

![autocomplete sentence](/_assets/GPT/GPT_00012.jpg)

> "and the only reason why i'm still alive today was to get a job at the hospital so that my mom could go back home to her parents for the rest the day so that she could"  
> -Carter's iPhone

