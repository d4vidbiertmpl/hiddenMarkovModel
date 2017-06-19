# hiddenMarkovModel

## Description
My Approach of an Part of Speech-Tagger using a Hidden Markov Model. The HMM is essentially working (probably waited too long to commit to GitHub ;-) ), but it definitely needs some more work. First of all it needs some refactoring, as I am not happy with the overall structure and the code is quite ugly at some places.  Secondly (and most important) big performance improvements need to be done. I worked with Python Lists and Dictionaries, which was quite convenient, but is way to slow for daily, “competitive” use. Using Numpy data structures will solve most of these problems. Moreover the code is calculating the trained probabilities each run from scratch and the matrices calculations can be done more cleverly.

## TODO:

<li> General refactoring
<li> Using numpy data structures
<li> Save trained emission and transition probabilities
<li> Better matrices organization
<li><li> Lazy access... ?

