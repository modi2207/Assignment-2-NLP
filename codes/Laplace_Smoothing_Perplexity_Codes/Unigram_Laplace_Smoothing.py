
import os
import math
import pandas as pd
import numpy as np

class UnigramPerplexity:

    def __init__(self):
        self.unigramCounts=pd.read_csv(os.getcwd()+"\\..\\results\\unigramCounts.csv")
        self.unigramCounts['Count After Smoothing']=np.NAN
        self.unigramCounts["Actual Probability"]=np.NAN
        self.unigramCounts["Probability After Smoothing"]=np.NAN
        self.probability={}  # dictionary contains probability after (applying laplace smoothing) for each unigram of corpus.
        self.V = len(self.unigramCounts.index)+43067  # vocabulary size for smoothing. We have taken total unique unigrams as vocabulary
        self.N = 0          # total tokens in corpus
        self.K = 1            # smoothing factor
        for i in range(0,len(self.unigramCounts)):
            self.N=self.N+self.unigramCounts.iloc[i,1]
        for i in range(0,len(self.unigramCounts)):
            self.probability[self.unigramCounts.iloc[i,0]]=self.unigramCounts.iloc[i,4]
        for i in range(0,len(self.unigramCounts)):
            self.probability[self.unigramCounts.iloc[i,0]]=(self.unigramCounts.iloc[i,1]+1)/(self.V+self.N);  # Probability of given unigram after smoothing
            self.unigramCounts.iloc[i,4]=(self.unigramCounts.iloc[i,1]+self.K)/(self.K*self.V+self.N)   # Probability of given unigram after smoothing
            self.unigramCounts.iloc[i,3]=self.unigramCounts.iloc[i,1]/self.N    # Probability of given unigram before smoothing
            self.unigramCounts.iloc[i,2]=((self.unigramCounts.iloc[i,1]+self.K)/(self.K*self.V+self.N))*(self.N)  # efective unigram count after smoothing
        self.unigramCounts.to_csv(os.getcwd()+"\\..\\results\\unigramCounts.csv",index=False)

        self.probability['UNKNOWN']=self.K/(self.N+self.K*self.V)
        self.findPerplexity()

    def getPerplexity(self,data):

        """
             find a perplexity for given sentence.
             :param data: stream of tokens represents sentence which is seprated by comma.
             :return: Perplexty of sentence
        """
        data=list(data.split(', '))
        print("data length: ",len(data))
        probSum=0
        for i in range(0,len(data)):
            if data[i]=='[END]\n':
                data[i]=str(data[i])
                data[i]=data[i][0:len(data[i])-1]
            probSum=probSum+math.log2(self.findUnigramProbability(data[i]))
        return pow(2,(-(probSum/len(data))))


    def findUnigramProbability(self,word):
        """

                find unigram  probability. i.e p(word)

                :param word: word for which we find probability
                :return: unigram probability
        """
        try:
            return self.probability[word]
        except KeyError as error:
            print("error occurred: ",word)
            return self.probability['UNKNOWN']





    def findPerplexity(self):
        """
               This method read line by line data from testing_data.txt and call getPerplexity() method for each line
               and append line & it's perplexity to pandas dataframe.
        """
        file=open('..\\data\\testing_data.txt','r',encoding="utf8")
        lines=file.readlines()
        unigramDF=[]
        for line in lines:
            perplexity=self.getPerplexity(line)
            unigramDF.append([line,perplexity])
        df=pd.DataFrame(unigramDF,columns=['Data','Perplexity'])
        df.loc[len(df), :] = {'Perplexity': df["Perplexity"].mean()}
        df.to_csv(os.getcwd()+"\\..\\results\\unigram_Perplexity_with_smoothing.csv",index=False)
unigramPerplexity=UnigramPerplexity()


