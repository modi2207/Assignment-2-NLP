import os
import math
import pandas as pd
import numpy as np

class QuadgramPerplexity:

    def __init__(self):
        """
        constructors initialize all necessary variables.
        """
        self.quadgramDF=pd.read_csv(os.getcwd()+"\\counts\\quadgram_counts.csv")
        self.quadgramDF['Count After Smoothing'] = np.NAN
        self.quadgramDF["Actual Conditional Probability"] = np.NAN
        self.quadgramDF["Conditional Probability After Smoothing"] = np.NAN
        self.biagramDF = pd.read_csv(os.getcwd() + "\\results\\biagram_counts.csv")
        self.unigramDF = pd.read_csv(os.getcwd() + "\\results\\unigramCounts.csv")
        self.traigramDF = pd.read_csv(os.getcwd() + "\\results\\traigram_counts.csv")
        self.V = len(self.unigramDF.index)  # vocabulary size for smooting. We have taken total unique unigrams as vocabulary
        self.K = 1    # smoothing factor
        self.N=0
        self.biagramCounts={}   # dictionary contains count for each bigram
        self.traigramCounts={}   # dictionary contains count for each trigram
        self.unigramCounts={}  # dictionary contains count for each unigram
        self.traigramProbability={}   # dictionary contains trigram probability (after applying laplace smoothing)for each trigram
        self.unigramProbability = {}   # dictionary contains unigram probability (after applying laplace smoothing) for each uniagram
        self.biagramProbability = {}  # dictionary contains biagram probability (after applying laplace smoothing) for each biagram
        self.quadgramProbability={}  # dictionary contains quadgram probability (after applying laplace smoothing) for each quadgram
        for i in range(0,len(self.unigramDF)):
            self.unigramCounts[self.unigramDF.iloc[i,0]]=self.unigramDF.iloc[i,1]
            self.unigramProbability[self.unigramDF.iloc[i,0]]=self.unigramDF.iloc[i,4]
            self.N=self.N+self.unigramDF.iloc[i,1]

        for i in range(0,len(self.biagramDF)):
            self.biagramCounts[self.biagramDF.iloc[i,0]]=self.biagramDF.iloc[i,1]
            self.biagramProbability[self.biagramDF.iloc[i, 0]] = self.biagramDF.iloc[i, 4]

        for i in range(0,len(self.traigramDF)):
            self.traigramCounts[self.traigramDF.iloc[i,0]]=self.traigramDF.iloc[i,1]
            self.traigramProbability[self.traigramDF.iloc[i, 0]] = self.traigramDF.iloc[i, 4]

        for i in range(0,len(self.quadgramDF)):
            c= self.traigramCounts[str([eval(self.quadgramDF.iloc[i,0])[0],eval(self.quadgramDF.iloc[i,0])[1],eval(self.quadgramDF.iloc[i,0])[2]])] if str([eval(self.quadgramDF.iloc[i,0])[0],eval(self.quadgramDF.iloc[i,0])[1],eval(self.quadgramDF.iloc[i,0])[2]]) in self.traigramCounts else 0
            self.quadgramDF.iloc[i,2]=((self.quadgramDF.iloc[i,1]+self.K)/(c+self.K*self.V))*c  # Effective count of biagram after smoothing
            self.quadgramDF.iloc[i,3]=(self.quadgramDF.iloc[i,1]/c)       # probabilty of given biagram in corpus (before smoothing)
            self.quadgramDF.iloc[i,4]=((self.quadgramDF.iloc[i,1]+self.K)/(c+self.K*self.V))    #  probability of given biagram after smoothing
            self.quadgramProbability[self.quadgramDF.iloc[i,0]]=self.quadgramDF.iloc[i,4]

        self.quadgramDF.to_csv("results\\quadgram_counts.csv",index=False)
        self.findPerplexity()
    def getPerplexity(self,data):
        """
        find a perplexity for given sentence.
        :param data: stream of tokens represents sentence which is seprated by comma.
        :return: Perplexty of sentence
        """
        data=list(data.split(', '))
        probSum=0
        for ind in range(0, len(data)):
            context = []
            prev_ind = max(-1, ind - 4)

            # consider at most 3 context word as we use quadgram for finding perplexity

            for prev in range(ind, prev_ind, -1):
                context.append(data[prev])

            context = list(reversed(context))


            if len(context) == 1:
                if context[0] in self.unigramProbability:
                    probSum += math.log2(self.unigramProbability[context[0]])
                else:
                    probSum+=math.log2(self.findUnknownProbability(context))
            elif len(context) == 2:
                if str(context) in self.biagramProbability:
                    probSum += math.log2(self.biagramProbability[str(context)])
                else:
                    probSum += math.log2(self.findUnknownProbability(context))
            elif len(context) == 3:
                if str(context) in self.traigramProbability:
                    probSum += math.log2(self.traigramProbability[str(context)])
                else:
                    probSum += math.log2(self.findUnknownProbability(context))
            else:
                if str(context) in self.quadgramProbability:
                    probSum += math.log2(self.quadgramProbability[str(context)])
                else:
                    probSum += math.log2(self.findUnknownProbability(context))

        probSum = abs(probSum)
        probSum /= len(data)
        return pow(2, probSum)

    def findUnknownProbability(self,context):
        """
               it finds probability of n-gram which are not present in respective n-gram probability dictionary
               :param context: n-gram (list of words)
               :return: probability
               """
        if len(context)==1:
            return self.K/self.K*self.V+self.N
        elif len(context)==2:
            c = self.unigramCounts[context[0]] if context[0] in self.unigramCounts else 0
            return self.K/(c+self.K*self.V)
        elif len(context)==3:
            c = self.biagramCounts[str([context[0],context[1]])] if str([context[0],context[1]]) in self.biagramCounts else 0
            return self.K/(c+self.K*self.V)
        else:
            c = self.traigramCounts[str([context[0], context[1],context[2]])] if str([context[0], context[1],context[2]]) in self.traigramCounts else 0
            return self.K / (c + self.K*self.V)

    def findPerplexity(self):
        """
        This method read line by line data from testing_data.txt and call getPerplexity() method for each line
        and append line & it's perplexity to pandas dataframe.
        """
        file=open('testing_data.txt','r',encoding="utf8")
        lines=file.readlines()
        quadgramPPDF=[]
        for line in lines:
            perplexity=self.getPerplexity(line)
            quadgramPPDF.append([line,perplexity])
        df=pd.DataFrame(quadgramPPDF,columns=['Data','Perplexity'])
        df.loc[len(df), :] = {'Perplexity': df["Perplexity"].mean()}
        df.to_csv("results\\quadgramPerplexity.csv")

quadgramPerplexity=QuadgramPerplexity()









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
