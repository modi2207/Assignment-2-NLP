import pandas as pd
import csv
import math

# load all csv file which contains n-gram conditional probabilities

bigram_csv = pd.read_csv('bigram_conditional_probabilities.csv')
unigram_csv = pd.read_csv('unigram_probabilities.csv')

# make dictionary for all this n-gram

bigram = {}
unigram = {}

# Store all n-gram conditional probabilities in dictionary list of words as key and probability as value

for index, row in bigram_csv.iterrows():
    text = eval(row['Bigram'])
    text = tuple(text)
    conditional_probability = float(row['Conditional Probability'])
    bigram[text] = conditional_probability

for index, row in unigram_csv.iterrows():
    text = row['Word']
    text = [text]
    text = tuple(text)
    conditional_probability = float(row['Probability'])
    unigram[text] = conditional_probability

with open('bigram_test_data_perplexity.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Test_Comments", "Perplexity"])

    with open('testing_data.txt', 'r', encoding="utf-8") as data:
        # Iterate through each line in the file

        total_perplexity = 0
        total_count = 0

        for line in data:
            # split each words of sentence
            words = line.split(', ')
            perplexity = 0
            N = len(words)
            for ind in range(0, len(words)):
                context = []
                prev_ind = max(-1, ind - 2)

                # consider at most 1 context word as we use bigram for finding perplexity

                for prev in range(ind, prev_ind, -1):
                    context.append(words[prev])

                context = list(reversed(context))

                # removing new line character at the end of sentence
                if context[-1] == '[END]\n':
                    context[-1] = '[END]'

                context = tuple(context)

                if len(context) == 1:
                    if context in unigram and unigram[context] != 0:
                        perplexity += math.log2(unigram[context])
                    else:
                        perplexity = "INFINITE"
                        break

                elif len(context) == 2:
                    if context in bigram and bigram[context] != 0:
                        perplexity += math.log2(bigram[context])
                    else:
                        perplexity = "INFINITE"
                        break

            if perplexity == "INFINITE":
                writer.writerow([line, perplexity])
                continue

            perplexity = abs(perplexity)
            perplexity /= N
            perplexity = pow(2, perplexity)

            total_perplexity += perplexity
            total_count += 1
            # store comments and corresponding perplexity
            writer.writerow([line, perplexity])

    # Store average perplexity
    Average_Perplexity = total_perplexity / total_count
    writer.writerow(['Average Perplexity (considering only finite values)', Average_Perplexity])


