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


if __name__ == "__main__":
    # Example sentence
    sentence = "pick the hot dog from the cabinet and place it on the counter"

    # Extract noun phrases
    noun_phrases = extract_noun_phrases(sentence)

    # Print the extracted noun phrases
    print(noun_phrases)
