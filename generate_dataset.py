"""
Generates dataset.csv for the Speech-to-Text NLP Prediction project.
Domain: Product / service review sentiment classification (Positive / Negative / Neutral).
This is the dataset the notebook and app are trained against.
"""

import random
import pandas as pd

random.seed(42)

subjects = [
    "the product", "this phone", "the delivery", "the service", "the app",
    "this laptop", "the customer support", "the packaging", "this headset",
    "the food", "the hotel room", "this course", "the movie", "the book",
    "the software", "this camera", "the battery life", "the website",
    "the staff", "this restaurant", "the shoes", "this jacket", "the game",
    "the seller", "the driver", "the flight", "this speaker", "the warranty",
    "the return process", "this watch"
]

positive_templates = [
    "I am really happy with {s}.",
    "{S} exceeded all my expectations.",
    "Absolutely loved {s}, would recommend to everyone.",
    "{S} works perfectly and looks great.",
    "Very satisfied with {s}, five stars.",
    "{S} is amazing, best purchase I made this year.",
    "Great experience with {s}, will buy again.",
    "{S} is fantastic and worth every rupee.",
    "I am impressed by the quality of {s}.",
    "{S} made my day, truly wonderful.",
    "Excellent, {s} arrived on time and works flawlessly.",
    "{S} is super comfortable and easy to use.",
    "So glad I chose {s}, it is outstanding.",
    "{S} is exactly what I needed, superb quality.",
    "Highly recommend {s} to anyone looking for quality.",
]

negative_templates = [
    "I am very disappointed with {s}.",
    "{S} was a complete waste of money.",
    "Terrible experience with {s}, would not recommend.",
    "{S} stopped working after a single day.",
    "Very unhappy with {s}, worst purchase ever.",
    "{S} is poorly made and broke quickly.",
    "Bad experience with {s}, never buying again.",
    "{S} did not match the description at all.",
    "I regret buying {s}, total waste.",
    "{S} was late, damaged, and useless.",
    "Awful, {s} does not work as promised.",
    "{S} is uncomfortable and cheaply built.",
    "So frustrated with {s}, it barely functions.",
    "{S} is not worth the price, very poor quality.",
    "Extremely dissatisfied with {s}, avoid it.",
]

neutral_templates = [
    "{S} is okay, nothing special.",
    "{S} arrived as expected, no issues.",
    "{S} does the job, average quality.",
    "It's fine, {s} works but could be better.",
    "{S} is neither good nor bad, just average.",
    "I have mixed feelings about {s}.",
    "{S} was delivered on the scheduled date.",
    "{S} is acceptable for the price.",
    "Not bad, not great, {s} is just okay.",
    "{S} functions as described, nothing more.",
    "{S} is a standard option, does what it says.",
    "I have no strong opinion about {s}.",
    "{S} is reasonable but unremarkable.",
    "{S} met the basic requirements, that's it.",
    "It's an ordinary experience with {s}.",
]

def fill(template, subject):
    return template.format(s=subject, S=subject[0].upper() + subject[1:])

rows = []
for subject in subjects:
    for t in positive_templates:
        rows.append((fill(t, subject), "Positive"))
    for t in negative_templates:
        rows.append((fill(t, subject), "Negative"))
    for t in neutral_templates:
        rows.append((fill(t, subject), "Neutral"))

df = pd.DataFrame(rows, columns=["text", "target"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# add a few duplicate/near-duplicate rows and one missing value to mimic real, slightly messy data
df.loc[len(df)] = [None, "Positive"]
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)

df.to_csv("dataset.csv", index=False)
print(df.shape)
print(df["target"].value_counts())
