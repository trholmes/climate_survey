import pandas as pd

# File to make basic selection standards

import json
f = open('selections.json')
selections = json.load(f)

bins = {}
vals = ["gender", "race", "lgbtq"]

for val in vals: bins[val] = []

for dataset in selections:

    fname = selections[dataset]["file"]
    with open(fname, newline='') as csvfile:
        responses = pd.read_csv(csvfile)
        questions = responses.columns.values

    for val in vals:
        n_q = selections[dataset][val]
        bins[val] += list(responses.iloc[:, n_q])

for val in vals:
    bins[val] = set(bins[val])

# Want to print out things like this
# selections[dataset]["gender"]: {"Female and Other": ["Female", "Nonbinary / Third Gender"],"Male": ["Male"]},

# Handle gender
# Male, Female and Nonbinary
str = "selections[dataset]['gender']: {"
str += "'Male': ['Male'], "
str += "'Female and Nonbinary': ["
for ans in bins['gender']:
    if ans != 'Male' and ans != 'Prefer not to disclose':
        str += "'%s', "%ans
str += "]},"
print(str)

# Handle race
# Only selected White/Caucasian, Other
str = "selections[dataset]['race']: {"
str += "'Only selected White/Caucasian': ['White/Caucasian'], "
str += "'Other': ["
for ans in bins['race']:
    if ans != 'White/Caucasian' and ans != 'Prefer not to disclose':
        str += "'%s', "%ans
str += "]},"
print(str)

# Handle sexual identity
# Only selected Heterosexual, Other
str = "selections[dataset]['lgbtq']: {"
str += "'Only selected Heterosexual': ['Heterosexual/Straight', 'Heterosexual/Straight, never think about this question'], "
str += "'Other': ["
for ans in bins['lgbtq']:
    if ans != 'Heterosexual/Straight' and ans != 'Straight' and ans != 'Heterosexual/Straight, never think about this question' and  ans != 'Prefer not to disclose':
        str += "'%s', "%ans
str += "]},"
print(str)

