"""
Required imports
----------------
"""

import os
import nltk
from math import log10,sqrt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import time
#debugging
import timeit
import json

""" 
Initializing all the required variables
---------------------------------------
corpusroot : str
    The root location of the corpus
tokenizer : RegexpTokenizer
    The Instance of RegexpTokenizer with custom Regex expression (r'[a-zA-Z]+') 
	which will create token of each word from our corpus file
stop_words  : set
    The set of all english stop words from nltk stopwords dictionary
stemmer : method
    The instance of PotterStemmer stem method to get the base word of
	the token
tf : dict
    The nested dictionary of term frequency of corpus. Count of Number
	of times a token has appeared in a document. 
	Format->{filename1: { token1: count1,..},..}
df : Counter
    The dictionary of Number of documents a token appeared.
	Format->{token1: Count1,...} 
tfidf_vectors : dict
	The dictionary of normalized TF-IDF vector.
    Format->{filename1:{token1:Weight1,..},..} 
postings_list : dict
    The dictionary of all toekns along with list of files and weights 
	where they appeared in which each element is in the form of 
	(document  d , TF-IDF weight  w. 
	Format->{token1:{filename1:Weight1,..},..} 
    
"""

corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
#stemmer = SnowballStemmer("english").stem
stemmer = PorterStemmer().stem
tf = {}
df = Counter()
tfidf_vectors = {}
postings_list = {}



def preProcess():
    """ Preprocess all the files from the corpus
    
    It reads the each file one at time and tokenize it, remove stop words, stem those token count tf and df
     
    Local Variables
    ---------------
    tokens : list, temp : list
        The temporary variable for storage
        
    Requirement
    -----------
        This method requires valid Directory name from where to read corpus.
        
    Outcome
    -------
        tf -> term frequency
        df -> document frequency
            
    """
    global df
    
    #Read files from the corpus directory in read mode
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        doc = doc.lower()

        # tokenizing all the words from the document
        tokens = tokenizer.tokenize(doc)

        # stopwords remove and stemming
        # case 1 time = 3.834928661815138
        temp = []
        append = temp.append
        for token in tokens:
            if token not in stop_words: 
                append(token)

        #Using map to map stemmer function to all temp list elemets at once and Typecating to list again
        tokens = list(map(stemmer, temp)) 

        # case 2 time = 6.202010461137888
        # tokens = list(map(lambda x: stemmer(x), filter(lambda x: x not in stop_words, tokens)))

        # Counting term frequency and storing in tf dict. 
        # Counter is inbuild function that Counts the element occurance in a list
        tf[filename] = Counter(tokens);
    
        # counting document frequency
        # converting tokens to set to remove duplicates which avoids multiple count in single document
        df += Counter(set(tokens))

def getidf(token: str) -> float:
    """ Calculates IDF - Inverse Document Frequency of a token
    
    Parameter
    ----------
        token : str
            Token whose idf needs to be calculated
    
    Requirement
    -----------
        tf -> to find toltal number of docs
        df of the token
    
    Returns
    -------
        float
            return calculated idf value or -1
    """
    
    # Return -1 if token is not present in df
    # else return calcuated idf value of token using :
    # log10(len(tf)/df[token]):
    #           len(tf) -> total doc
    #           df[token] -> df of the token
    return -1 if df[token] == 0 else log10(len(tf)/df[token])


def create_tfidf_vectors():
    """ Calculates TF-IDF Vector of the token'
    
    It calculates raw tf-idf vector and the normalize those value
   
   Local Variables
    ---------------
        vector_list  : dict
            The dictionary of unnormalized TF-IDF vector.
            Format->{filename1:{token1:Weight1,..},..} 
        vector_magnitude : dict
            The dictionary of magnitude of a document of  unnormalized TF-IDF 
            weights.
            Format->{filename1: magnitude1,...}
    
    Requirement
    -----------
        tf -> to find toltal number of docs
        getidf(token) -> method
        
    Outcome
    -------
        tfidf_vectors -> Normalized TF-IDF vector
    """
    vector_list = {}
    vector_magnitude = {}
    for file,tokens in tf.items():
        
        """calculates raw tf-idf
        For a given dict of tokens we extract keys using tokens.keys()
        Using Lambda we calculate tf-idf for each token in the tokens dict
        and then return a key:value pair dict
        where key -> token name , value -> un normalized tf-idf and store in vector_list"""
        vector_list[file] = dict(map(lambda token : (token,(1+log10(tokens[token]))*getidf(token)) ,tokens.keys()))
        
        """calculates file magnitude
        Form the calculated vector_list using vector_list[file].values() 
        Using Lambda we calculate magnitude of the each document
        and then return a key:value pair dict
        where key -> file name , value -> magnitude of the file"""
        vector_magnitude[file] = (sqrt(sum(map(lambda value : value * value ,vector_list[file].values()))))
        
        tfidf_vectors[file] = Counter()
        
        #normalization of each token with respect document in which they are present
        for token in vector_list[file]:
            tfidf_vectors[file][token] = vector_list[file][token] / vector_magnitude[file]

def create_postings_list():
    """Creates posting list of all tokens
    
    Requirement
    -----------
        df -> to create empty posting_list with token as the key and value of type dict
        tfidf_vectors -> to get normalized value of the token and filename
    """
    for token in df: postings_list[token]={}
    for file,tokens in tfidf_vectors.items():
        for token in tokens:
            postings_list[token].update({file:tfidf_vectors[file][token]})

def find_common(doc_dict: dict) -> dict:
    """Finds the common doc from top10 posting list
    
    Parameter
    ---------
    doc_dict : dict
        Top10 postinglist dict
    
    Outcome
    -------
        dictionary of all common docs
    """
    #Assigning counter to similar doc so that everytime a same doc is entered it will increment it value by 1
    simmilar_doc = Counter()
    
    #reading all tokens from doc
    for token in doc_dict:  
        #Extracting file name from doc_dict and assiging it to similar doc for incrementing
        simmilar_doc += Counter(doc_dict[token].keys())    
    #return all comon docs i.e
    #checking wheather count of any file in similar_doc == to len of tokens in doc_dict
    return {doc_name for doc_name,count in dict(simmilar_doc).items() if (count == len(doc_dict))}

def sort_lists_by_value(token: dict,length = None) -> dict:
    """Sorting dictionary by values in descending order
    
    Parameter
    ---------
    token : dict
        The dictionary which need to be sorted
        
    lenght : int (optional)
        States the lenght of the return dictionary
        default : none -> returnes all values
    
    Return
    ------
    dict
        Return sorted dictionary of specific length if specified
    """
    
    if length is None:        
        return dict(sorted(token.items(), key=lambda kv: kv[1], reverse=True))
    return dict(sorted(token.items(), key=lambda kv: kv[1], reverse=True)[:length])


def query(qstring: str) -> tuple:
    """Calculates cosine-similarity of the query to the document and return the most relevant document based on score
    
    Parameter
    ---------
    qstring : str
        Query string whose relevant docs need to be found
    
    Local Variables
    ---------------
    similar_doc : dict
    temp : dict
    sim_score : dict
    top_ten_list = dict
        The sorted dictionary by weight of top 10 weight for each token 
        from postings_list. 
        Format->{token1:{filename1:Weight1,..},..} 
    Returns
    -------
        tupe -> (Filename, document_score)
    """
    #initializing 
    similar_doc = {}
    temp = {}
    sim_score = {}
    top_ten_list = {}
    #tokenizing query
    qtokens = tokenizer.tokenize(qstring.lower())
    
    #removing stopwords from qtoken, stemming and counting the occurance ofthe words
    qtokens = Counter(list(map(stemmer, [token for token in qtokens if token not in stop_words])))
    
    #calculating weight of each token using 1+log10(no of occurance)
    qvector_list = dict(map(lambda token:(token,1+log10(qtokens[token])),qtokens.keys()))    
    
    validtokens = []
    for qtoken in qvector_list:
        if qtoken not in df: #checking if token exist in df. Ignoring it if not present
            continue
        #creating top10 from postinglist using qtokens and soring it
        #sort_lists_by_value will return descinding order 10 sorted element list
        top_ten_list[qtoken] = sort_lists_by_value(postings_list[qtoken],10)
        validtokens.append(qtoken)
    
    """If there is not document for any token in the query return none"""
    if len(top_ten_list.keys()) == 0:
        return None,0
    
    #calculating magnitute of the qvectors for normalization
    qmagnitude =  (sqrt(sum(map(lambda kv : (kv[1] * kv[1])*qtokens[kv[0]] ,qvector_list.items()))))
    
    #normalizing each token in qvectorlist
    for token in qvector_list:            
            qvector_list[token] = qvector_list[token] / qmagnitude     
    
    #finding all the similar doc from all the tokens in top_ten_list
    similar_doc = find_common(top_ten_list)    
    
    #finding cosin-similarity
    for file in tfidf_vectors:
        sim_score[file] = 0
        temp_score = 0
        for token in validtokens:
            if file in top_ten_list[token]:
                sim_score[file] += qvector_list[token]*tfidf_vectors[file][token]
                #print('i am if ' + token + " " +file+ " " + str(sim_score[file])) 
            else:
                upper_bond = list(top_ten_list[token].values())[-1]
                sim_score[file] += qvector_list[token]*upper_bond
                #print('i am if ' + token + " " +file+ " " + str(sim_score[file])) 
    
    #print(json.dumps(sort_lists_by_value(sim_score), indent=2))
    #Sorting and geting highest score
    sim_name,sim_score = next(iter(sort_lists_by_value(sim_score,1).items()))
    
    """Checking If a document's actual score is better than or equal to the sims scores of all other documents, it is returned as the query answer or if there isint any match returns fetch more"""
    if sim_name in similar_doc:
        return sim_name, sim_score
    else:
        return ("fetch more",0)


    
def getweight(filename: str,token: str) -> float:  
    """it return the normalized tfidf weight of a specified token
    
    Parameter
    ---------
    filename :  str
    token : str
    
    Returns
    -------
    normalized weight of token from specified file
    """
    return 0 if token not in tfidf_vectors[filename] else tfidf_vectors[filename][token]


def main():
    start = time.time()

    preProcess()
    create_tfidf_vectors()
    create_postings_list()	
    print("(%s, %.12f)" % query("health insurance wall street"))
    #(2012-10-03.txt, 0.033877975254)

    print("(%s, %.12f)" % query("particular constitutional amendment"))
    #(fetch more, 0.000000000000)

    print("(%s, %.12f)" % query("terror attack"))
    #(2004-09-30.txt, 0.026893338131)

    print("(%s, %.12f)" % query("vector entropy"))
    #(None, 0.000000000000)

    print("%.12f" % getweight("2012-10-03.txt","health"))
    #0.008528366190

    print("%.12f" % getweight("1960-10-21.txt","reason"))
    #0.000000000000

    print("%.12f" % getweight("1976-10-22.txt","agenda"))
    #0.012683891289

    print("%.12f" % getweight("2012-10-16.txt","hispan"))
    #0.023489163449

    print("%.12f" % getweight("2012-10-16.txt","hispanic"))
    #0.000000000000

    print("%.12f" % getidf("health"))
    #0.079181246048

    print("%.12f" % getidf("agenda"))
    #0.363177902413

    print("%.12f" % getidf("vector"))
    #-1.000000000000

    print("%.12f" % getidf("reason"))
    #0.000000000000

    print("%.12f" % getidf("hispan"))
    #0.632023214705

    print("%.12f" % getidf("hispanic"))
    #-1.000000000000
    end = time.time()
    print(end - start)
    
#(None, 0.000000000000)

if __name__ == "__main__":
    main()
