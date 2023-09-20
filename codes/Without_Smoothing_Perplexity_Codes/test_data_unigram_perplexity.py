import pandas as pd
import csv
import math

# load all csv file which contains n-gram conditional probabilities

unigram_csv = pd.read_csv('unigram_probabilities.csv')

# make dictionary for all this n-gram

unigram = {}

# Store all n-gram conditional probabilities in dictionary list of words as key and probability as value

for index, row in unigram_csv.iterrows():
    text = row['Word']
    text = [text]
    text = tuple(text)
    conditional_probability = float(row['Probability'])
    unigram[text] = conditional_probability


with open('unigram_test_data_perplexity.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Test_Comments", "Perplexity"])

    with open('testing_data.txt', 'r', encoding="utf-8") as data:
        data = data.readlines()
        # Iterate through each line in the file

        total_perplexity = 0
        total_count = 0

        for line in data:
            # split each words of sentence
            words = line.split(', ')
            perplexity = 0;
            N = len(words)

            for ind in range(0, len(words)):
                context = []
                context.append(words[ind])

                # removing new line character at the end of sentence
                if context[-1] == '[END]\n':
                    context[-1] = '[END]'
                context = tuple(context)

                if context in unigram and unigram[context] != 0:
                    perplexity += math.log2(unigram[context])
                else:
                    print(context)
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
    Average_Perplexity = total_perplexity/total_count
    writer.writerow(['Average Perplexity', Average_Perplexity])




