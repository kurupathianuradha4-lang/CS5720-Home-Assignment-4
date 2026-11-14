# CS5720-Home-Assignment-4

Course: CS5720 Neural Networks and Deep Learning, Fall 2026 University: University of Central Missouri Student Name: Anuradha Kurupathi Student ID: 700781806

Overview

This repository contains my solutions for Home Assignment 4. Part I covers short-answer questions on sequence-to-sequence models and attention, self-attention, multi-head attention, and the Transformer architecture. Part II covers an NLP preprocessing pipeline with NLTK (Q1), named entity recognition with spaCy (Q2), and a from-scratch implementation of scaled dot-product attention in NumPy (Q3).

Repository Contents
q1_nlp_preprocessing.py — tokenization, stopword removal, and stemming with NLTK
q2_ner_spacy.py — named entity extraction with spaCy
q3_attention.py — scaled dot-product attention using NumPy
README.md — this file
Part I: Short-Answer Questions
Q1: Sequence-to-Sequence Models and Attention

a. The encoder reads the source sentence token by token and compresses it into hidden states / a context vector. The decoder uses that representation to generate the target sentence one token at a time, conditioned on the tokens it has already produced.

b. The whole source sentence must fit into one fixed-length vector. This is a bottleneck: for long sentences the vector cannot hold all the information, early words get forgotten, and translation quality drops.

c. Attention lets the decoder look at all encoder hidden states at every decoding step. It computes a weight for each source position and builds a context vector as the weighted sum of the encoder states, so the decoder can focus on the source words most relevant to the word it is generating.

Q2: Attention Mechanisms

a. Hard attention selects one (or a few) positions, which is a discrete, non-differentiable choice. Soft attention gives every position a probability weight and takes a weighted average. Soft attention is differentiable, so gradients flow through the weights and it can be trained with ordinary backpropagation.

b. With s = [1, 0]:

e₁ = sᵀh₁ = 1·1 + 0·0 = 1
e₂ = sᵀh₂ = 1·0 + 0·1 = 0
e₃ = sᵀh₃ = 1·1 + 0·1 = 1

c. Softmax: e¹ = 2.718, e⁰ = 1, e¹ = 2.718, sum = 6.437

Weight	Value
α₁	0.422
α₂	0.155
α₃	0.422

Context vector = 0.422·[1,0] + 0.155·[0,1] + 0.422·[1,1] = [0.844, 0.577]

Q3: Self-Attention in Transformers

a. Self-attention lets every token in a sequence attend to every other token in the same sequence, so each token's representation is updated with context from the whole sentence. In Seq2Seq encoder-decoder attention, the queries come from the decoder and the keys/values come from the encoder (two different sequences). In self-attention, Q, K, and V all come from the same sequence.

b. Query (Q): what the current token is looking for. Key (K): what each token offers, matched against the queries. Value (V): the information that is actually passed on. QKᵀ gives similarity scores, softmax turns them into weights, and the output is the weighted sum of V.

c. Every token connects directly to every other token in a single step, so distance does not matter. Example: in "The animal didn't cross the street because it was too tired", the word "it" can attend strongly to "animal" even though several words separate them.

Q4: Multi-Head Attention

a. Multi-head attention runs several attention operations in parallel, each with its own learned Q/K/V projections. Multiple heads let the model capture different kinds of relationships at once (syntax, coreference, nearby vs. distant words), which a single head would average together.

b. d_head = d_model / h = 512 / 8 = 64

c. The outputs of all heads are concatenated (8 × 64 = 512) and passed through a final learned linear projection Wᴼ to produce the multi-head attention output.

Q5: Transformer Architecture

a. Self-attention has no sense of order, so the model would see a bag of words. Positional encodings are added to the embeddings to give each token its position in the sequence, so "dog bites man" is different from "man bites dog".

b. Masked self-attention hides future positions (set to −∞ before softmax), so position i can only attend to positions ≤ i. During autoregressive training the full target sequence is fed in at once, so without the mask the model could see the word it is supposed to predict and would fail at inference time.

c. RNNs process tokens one after another, so they cannot be parallelized across time. Transformers process all positions at the same time, so they train much faster on GPUs and handle long-range dependencies well (advantage). During autoregressive generation, however, tokens must still be produced one at a time, and attention cost grows quadratically with sequence length (limitation).

Part II: Programming Questions
Q1: NLP Preprocessing Pipeline
Wrote preprocess() that tokenizes the sentence with word_tokenize
Removed English stopwords (NLTK list, compared in lowercase) and punctuation
Applied the Porter stemmer to each remaining word
Input: "NLP techniques are used in virtual assistants like Alexa and Siri."
<!-- Replace with your actual output after running the script -->

Result:

Stage	Output
Original Tokens	['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Without Stopwords	['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
Stemmed Words	['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']

Stemming vs. lemmatization: Stemming chops suffixes with simple rules and can produce non-words. Lemmatization uses a vocabulary and part of speech to return a real dictionary form. For "running", stemming gives "run", while lemmatization gives "run" when treated as a verb and "running" when treated as a noun.

Stopwords: Removing them is useful for bag-of-words, TF-IDF, and search because it shrinks the vocabulary and drops low-information words. It can be harmful when small words carry meaning, e.g. sentiment analysis ("not good" becomes "good"), question answering, and machine translation.

Q2: Named Entity Recognition with spaCy
Loaded the en_core_web_sm spaCy model
Ran it on the given sentence and looped over doc.ents
Printed entity text, label, and start/end character positions
<!-- Replace with your actual output after running the script -->

Result:

Entity	Label	Start	End
LeBron James	PERSON	0	12
NBA	ORG	22	25
the United States	GPE	36	53
Laker	ORG	57	62
NBA Final MVP	ORG	71	84
2020	DATE	88	92

NER vs. POS tagging: POS tagging labels every token with its grammatical role (noun, verb, etc.). NER finds and classifies only spans that name real-world entities (PERSON, ORG, GPE, DATE), and an entity can span several tokens.

Applications: (1) Financial news: extracting company names, people, and amounts to track events. (2) Search engines: recognizing entities in queries (e.g. "weather in Paris") to return relevant results.

Q3: Scaled Dot-Product Attention
Computed QKᵀ, divided by √d (d = 4, so √d = 2), applied a row-wise softmax, and multiplied by V
Used the given Q, K, V test inputs

Result:

Attention weights	Output
[[0.731, 0.269], [0.269, 0.731]]	[[2.076, 3.076, 4.076, 5.076], [3.924, 4.924, 5.924, 6.924]]

Why divide by √d? Dot products grow with the key dimension, which pushes softmax into saturated regions where one weight is near 1 and the rest near 0, giving tiny gradients. Dividing by √d keeps the scores in a stable range.

How self-attention helps: Each word computes attention weights over all other words, so its representation is enriched by the most relevant words (e.g. a pronoun attends to its referent, a verb to its subject), regardless of distance.

How to Run
Install the requirements: pip install numpy nltk spacy
Download the spaCy model: python -m spacy download en_core_web_sm
Run python q1_nlp_preprocessing.py for the preprocessing output (downloads NLTK data on first run).
Run python q2_ner_spacy.py for the named entities.
Run python q3_attention.py for the attention weights and output.
