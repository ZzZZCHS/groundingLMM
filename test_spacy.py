import spacy

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

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

# Example sentences
sentence1 = "pick the apple from the counter and place it in the sink"
sentence2 = "John found it in the box."

# Extract direct object phrases
direct_object_phrases1 = extract_direct_object_phrases(sentence1)
direct_object_phrases2 = extract_direct_object_phrases(sentence2)

print("Direct Object Phrases for sentence 1:", direct_object_phrases1)
print("Direct Object Phrases for sentence 2:", direct_object_phrases2)
