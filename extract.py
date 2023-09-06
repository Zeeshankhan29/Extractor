import spacy

# Load the NER model
nlp = spacy.load("en_core_web_sm")

# Create a text string with a menu combination
text = "Chicken Biriyani with Garlic Naan - Rs. 200"

# Create a document object from the text string
doc = nlp(text)

# Find all the named entities in the document
ents = doc.ents

# Print the named entities and their labels
for ent in ents:
    print(ent.text, ent.label_)

