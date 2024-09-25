import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Define a function to extract noun phrases
def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if not any(token.pos_ == "PRON" for token in chunk):
            noun_phrases.append(chunk.text)
    return noun_phrases


def extract_direct_object_phrases(sentence):
    # Parse the sentence
    doc = nlp(sentence)
    
    # Store the direct object phrases
    direct_object_phrases = []
    
    # Iterate over tokens in the sentence
    for token in doc:
        # Check if the token is a direct object and not a pronoun
        if token.dep_ == "dobj" and token.pos_ != "PRON":
            # Find the noun chunk that contains this direct object
            for chunk in doc.noun_chunks:
                if token in chunk:
                    direct_object_phrases.append(chunk.text)
                    break
    
    return direct_object_phrases




if __name__ == "__main__":
    # Example sentence
    sentence = "pick the hot dog from the cabinet and place it on the counter"

    # Extract noun phrases
    noun_phrases = extract_noun_phrases(sentence)

    # Print the extracted noun phrases
    print(noun_phrases)
