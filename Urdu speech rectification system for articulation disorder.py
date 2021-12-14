import nltk
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.util import ngrams
import csv
import pandas as pd
from nltk.tokenize import word_tokenize
import itertools
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
import nltk
#nltk.download('punkt')


##################Data lena start##########################
with open('Sentence-DS-400.csv', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    
    count = 0
    fsa = []
    
    for row in reader:
        count = count + 1
        fsa.append(row['data'])
        if count > 401:
            break

#Language model

data = ''
for ele in fsa: 
    data += ele
    
unis_all = word_tokenize(data)
unis = set(unis_all)

bis = list(ngrams(data.split(), 2))

cols = rows = len(unis)
arr = [[0.0]*cols]*rows

df = pd.DataFrame(arr)
df.columns = unis
df.index = unis

v = len(unis)

dfp = df.copy(deep=True)


for b in bis:
   uni1 = b[0]
   uni2  = b[1]
   df.at[uni1,uni2] = df.at[uni1,uni2] + 1
   dfp.at[uni1,uni2] = (df.at[uni1,uni2] + 1) / (unis_all.count(uni1) + v)
   
  

def createBigram(text):
   listOfBigrams = []
   bigramCounts = {}
   unigramCounts = {}
   for i in range(len(data)-1):
      if i < len(data) - 1 and data[i+1]:

         listOfBigrams.append((data[i], data[i + 1]))

         if (data[i], data[i+1]) in bigramCounts:
            bigramCounts[(data[i], data[i + 1])] += 1
         else:
            bigramCounts[(data[i], data[i + 1])] = 1

      if data[i] in unigramCounts:
         unigramCounts[data[i]] += 1
      else:
         unigramCounts[data[i]] = 1
   return listOfBigrams, unigramCounts, bigramCounts

#ye sentence ki probability chec ki hai

def calcBigramProb(listOfBigrams, unigramCounts, bigramCounts):
   
    listOfProb = {}
    for bigram in listOfBigrams:
     try:
        word1 = bigram[0]
        word2 = bigram[1]
     
        listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
       
     except Exception:
         listOfProb[bigram] = 0
         continue   
    return listOfProb
   
    


if __name__ == '__main__':
    
    #it requires data in list
    data = unis_all
    listOfBigrams, unigramCounts, bigramCounts = createBigram(data)

    #inputSent="کوئلے تے آگ جلتی ہے"
    #inputSent="میں گاڑی تلاؤں گا"
    #inputSent="مجھے بھوک لدی تھی"
    #inputSent="مچھلی جل تی رانی ہے"
    #inputSent="مجھے کریلے پتند ہیں"
    #inputSent="مجھے ٹیکے  شے ڈر لگتا ہے"
    #inputSent="آج سالن تیا پکا گے"
    #inputSent="میں نے برف پلتی دیکھی جے"
    #inputSent="میری جیب خلچ کم ہے"
    #inputSent="ٹوٹی بند ترنی چاہیئے"
    #inputSent="گینڈا موتا جانور ہے"
    #inputSent="پھولوں سے توشبو آتی ہے"
    #inputSent="گھنٹی بد رہی ہے"
    #inputSent="ٹافی تھانی ہے"
    #inputSent="مجھے گرمیاں پتند ہیں "
    #inputSent="کنواں تل رہا ہے "
    #inputSent="تارے رات تو چمکتے ہیں"
    #inputSent="بادل گرش رہے ہیں "
    #inputSent="سگریٹ پینا اچھی بات ہے"
    inputSent="موبائل پل گیم لگادیں"
    
    
    
    splt=inputSent.split()
    outputProb1 = 1
    bilist=[]
    bigrm=[]

    for i in range(len(splt) - 1):
        if i < len(splt) - 1:
            bilist.append((splt[i], splt[i + 1]))
    
    correct_sentence = []
    ccb = []
    inputsent_Biprob = (calcBigramProb(bilist, unigramCounts, bigramCounts))
    
    print(inputsent_Biprob)
    stopword = ["ہے", "یہ", "وہ"]
    f = True
    skip = ""
    for b, p in inputsent_Biprob.items():
        print(b,p)
        ccb = b
        try:
            if(b[0] not in stopword):
                if( p < 0.0001):
                 
                    uni1 = b[0]
                    uni2 = b[1]
                    
                    if(skip != uni1):
                        probs = dfp.loc[uni1]
                        ccb = tuple([uni1, probs.nlargest(1).index[0]])
                        skip = uni2
                        print(ccb)
                    else:
                        skip = ""
            else:
                uni1 = b[0]
                uni2 = b[1]
        except Exception:
            continue
                
                
        if f:
            correct_sentence.append(ccb[0])
            f = False
        correct_sentence.append(ccb[1])

    sentence = ''
    for c in correct_sentence:
        sentence += ' '+c
    print(sentence)
    
    reference = ['ٹافی کھانی ہے'.split()]
    candidate = sentence.split()


    score = sentence_bleu(reference, candidate, weights=(1,0,0,0))
    print(score)
