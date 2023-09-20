import os
import math
import pandas as pd
import numpy as np

class BiagramPerplexity:

    def __init__(self):
        """
        constructors initialize all necessary variables.
        """
        self.biagramDF=pd.read_csv(os.getcwd()+"\\..\\results\\biagram_counts.csv")
        self.biagramDF['Count After Smoothing'] = np.NAN
        self.biagramDF["Actual Conditional Probability"] = np.NAN
        self.biagramDF["Conditional Probability After Smoothing"] = np.NAN
        self.unigramDF = pd.read_csv(os.getcwd()+"\\..\\results\\unigramCounts.csv")
        self.V = len(self.unigramDF.index)+43067  # vocabulary size for smooting. We have taken total unique unigrams as vocabulary
        self.K = 3    # smoothing factor
        self.N=0
        self.unigramCounts={}  # dictionary contains count for each unigram
        self.biagramProbability={}   # dictionary contains biagram probability (after applying laplace smoothing) for each biagram
        for i in range(0,len(self.unigramDF)):
            self.unigramCounts[self.unigramDF.iloc[i,0]]=self.unigramDF.iloc[i,1]
            self.N=self.N+self.unigramDF.iloc[i,1]


        for i in range(0,len(self.biagramDF)):
            try:

                c= self.unigramCounts[eval(self.biagramDF.iloc[i,0])[0]] if eval(self.biagramDF.iloc[i,0])[0] in self.unigramCounts else 0
                self.biagramDF.iloc[i,2]=((self.biagramDF.iloc[i,1]+self.K)/(c+self.K*self.V))*c  # Effective count of biagram after smoothing
                self.biagramDF.iloc[i,3]=(self.biagramDF.iloc[i,1]/c)       # probabilty of given biagram in corpus (before smoothing)
                self.biagramDF.iloc[i,4]=((self.biagramDF.iloc[i,1]+self.K)/(c+self.K*self.V))    #  probability of given biagram after smoothing
                self.biagramProbability[self.biagramDF.iloc[i,0]]=self.biagramDF.iloc[i,4]
            except Exception as err:
                print(err)
                print("bigram: ",self.biagramDF.iloc[i,0])

        self.biagramDF.to_csv(os.getcwd()+"\\..\\results\\biagram_counts_additive.csv",index=False)
        self.unigramCounts['UNKNOWN']=0
        self.findPerplexity()
    def getPerplexity(self,data):
        """
        find a perplexity for given sentence.
        :param data: stream of tokens represents sentence which is seprated by comma.
        :return: Perplexty of sentence
        """
        data=list(data.split(', '))
        #print(len(data))
        probSum=0
        for i in range(0,len(data)):
            if i>0:
                if data[i] == '[END]\n':
                    data[i] = str(data[i])
                    data[i] = data[i][0:len(data[i]) - 1]
                probSum=probSum+math.log2(self.findProbability(data[i],data[i-1]))
            else:
                probSum=math.log2(self.findProbability(data[i],None))
        return pow(2,(-(probSum/len(data))))

    def findProbability(self,word,prev):
        """

        find conditinal probability. i.e p(word/prev)

        :param word: target word for which we find probability
        :param prev: given word
        :return: conditional probability
        """
        if prev is None:  # it is for first word of sentence for which no base word.
            return (self.unigramCounts[word]+self.K)/(self.K*self.V+self.N) if word in self.unigramCounts else (self.unigramCounts['UNKNOWN']+self.K)/(self.K*self.V+self.N)
        else:

            try:
                print("got word")
                return self.biagramProbability[str([prev,word])]
            except KeyError as error:
                print("error occurred: ",(prev,word))
                c = self.unigramCounts[prev] if prev in self.unigramCounts else 0
                return 1 / (c + self.K*self.V)
    def findPerplexity(self):
        """
        This method read line by line data from testing_data.txt and call getPerplexity() method for each line
        and append line & it's perplexity to pandas dataframe.
        """
        file=open('..//data//testing_data.txt','r',encoding="utf8")
        lines=file.readlines()
        biagramPPDF=[]
        for line in lines:
            perplexity=self.getPerplexity(line)
            biagramPPDF.append([line,perplexity])
        df=pd.DataFrame(biagramPPDF,columns=['Data','Perplexity'])
        df.loc[len(df), :] = {'Perplexity': df["Perplexity"].mean()}
        df.to_csv(os.getcwd()+"\\..\\results\\biagram_Perplexity_with_smoothing_additive.csv")

biagramPerplexity=BiagramPerplexity()









