import pandas as pd

from bs4 import BeautifulSoup

def printVersion(name, module):
    print(name, module.__version__)

def printFeature(datafame, id, featureName):
    print(dataframe.ix[id][featureName])
    
def readFile(path, header, delimiter, quoting):
    return pd.read_csv(path, header=header, delimiter=delimiter, quoting=quoting)

def outputCSVFile(file, name, index, quoting):
    file.to_csv('data' + '/'+ name + '.csv', index=index, quoting=quoting)