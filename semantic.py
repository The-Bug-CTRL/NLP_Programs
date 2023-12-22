# semantic.py

import spacy

# Load the small-sized English NLP model
nlp_sm = spacy.load("en_core_web_sm")

# Process the text with the small model
doc_sm = nlp_sm("Andrew is smart")

# Print the similarity between the words
print("Similarity between cat and monkey:", doc_sm[0].similarity(doc_sm[1]))
print("Similarity between cat and banana:", doc_sm[0].similarity(doc_sm[2]))
print("Similarity between monkey and banana:", doc_sm[1].similarity(doc_sm[2]))

# note about differences here
print('''\nNote: I noticed that when using 'en_core_web_sm' its differs from en_core_web_md
in the way that md is bigger and also takes more ram and its use more detail whereas sm 
is smaller faster and is just quicker but more light weight''')

# Example with your own words
example_doc_sm = nlp_sm("Andrew is smart")
# Print similarity or do whatever analysis you want
print("Similarity within your example with 'en_core_web_sm':", example_doc_sm[0].similarity(example_doc_sm[1]))

