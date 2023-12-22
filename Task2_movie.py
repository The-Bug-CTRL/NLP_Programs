import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_md")

def get_most_similar_movie(user_description, movie_descriptions):
    # Tokenize and get the vector for the user's description
    user_vector = nlp(user_description).vector

    # Calculate cosine similarity between user's vector and each movie vector
    similarity_scores = []
    for movie_description in movie_descriptions:
        movie_vector = nlp(movie_description).vector
        similarity_score = cosine_similarity([user_vector], [movie_vector])[0][0]
        similarity_scores.append(similarity_score)

    # Find the index of the most similar movie
    most_similar_index = np.argmax(similarity_scores)

    # Return the title of the most similar movie
    return f"Movie {chr(ord('A') + most_similar_index)}"

# Read movie descriptions from the movies.txt file
with open('movies.txt', 'r') as file:
    movie_descriptions = [line.split(':')[1].strip() for line in file]

# User's description for "Planet Hulk"
user_description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator."

# Get the most similar movie
recommended_movie = get_most_similar_movie(user_description, movie_descriptions)

print(f"The recommended movie to watch next is: {recommended_movie}")
