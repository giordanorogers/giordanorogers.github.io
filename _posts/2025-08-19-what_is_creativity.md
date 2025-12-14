---
title: 'On creativity'
date: 2025-08-18
permalink: /posts/2025/08/what_is_creativity/
tags:
  - philosophy
  - creativity
  - computation
---


I define the following variables:
```python
x = 2
y = 6
```
Does the information that x and y are both even exist?

In one sense, yes.
The fact that x is 2 and y is 6 logically entails that both are even.
This is *implicit information*.
It exists in the sense that any reasoning system with the rules of arithmetic could derive it.

But in another sense, no.
The computer itself doesn't *know* they're even unless we encode that knowledge.
All it stores are the numbers 2 and 6 in an address associated with the variables x and y.
The property "evenness" only exists when checked, represented, and stored.

This is the distinction between **potential information** and **actualized information**.

Now I write a function and check the condition:
```python
def both_even(x,y):
	return x % 2 == y % 2 == 0
both_even_x_y = both_even(x,y)
```
Now the information that x and y are both even exists.

I've taken an implicit relation and made it explicit by constructing a new variable that records the result.
This information is not *logically* new.
It is a *representation of latent information* in an accessible form.

But this example isn't obviously relevant to real-world creativity.
Let's get more abstract, and show that this principle generalizes beyond numbers to attributes as well.

I define variables "Alice" and "Bob" with the following qualities:
```python
Alice = {"tall", "rich", "fast"}
Bob = {"short", "rich", "slow"}
```
The information that *Alice and Bob are both rich* only exists implicitly.
To make it explicit, I have to check it and store it.
```python
def both_rich(x,y):
	return "rich" in x and "rich" in y
both_rich_Alice_Bob = both_rich(Alice, Bob)
```

Is this checking and storing process creative?
Let's define a creative act as one that is novel and valuable.
It's a definition that evokes what we point to when we claim *creation* in both science and art.
In practice, creative acts, such as making a painting or a song, involve combining existing elements into something new, rather than generating wholly new objects.

The novelty of my `both_rich_Alice_Bob` representation is that I've written a new boolean value to an address in memory.
The value of it is that I've transformed an implicit structure into an explicit, accessible state.
 I now no longer have to run a function to check its truth.
 I can simply look it up in memory.
So yes, this is creative.

Everything possible is derivable from the properties of the universe.
Creativity doesn't mean magically conjuring up something from nothing.
It's the act of encoding implicit structure into an explicit representation.