# hiddenMarkovModel

## Description
My Approach implementing a Part-of-Speech (PoS) Tagger in Python from scratch using a Hidden Markov Model. The HMM is essentially working (probably waited too long to commit to GitHub ;-) ), but it definitely needs some more work. First of all some refactoring need to be done, as I am not happy with the overall structure, also the code is quite ugly at some places. Secondly (and most important) big performance improvements need to be done. I worked with Python lists and dictionaries, which was quite convenient, but is way to slow for daily, “competitive” use. Using "Numpy" data structures will solve most of these problems. Besides that, the trained emission and transition probabilities get recalculated every run, which is quite unnecessary. In addition the matrix multiplications can be done way more cleverly i think.

## TODO:

<li> General refactoring
<li> Using numpy data structures
<li> Save trained emission and transition probabilities
<li> Better matrices organization
<li> => Lazy loading/access... ? 

