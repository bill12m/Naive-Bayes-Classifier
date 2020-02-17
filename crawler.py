###Used this code to generate my data sets###
fromm bs4 import BeautifulSoup
import requests
import pandas as pd

import subprocess as sp
sp.call('clear', shell = True)

response = requests.get("https://en.wikipedia.org/wiki/Cyber_Branch_(United_States_Army)")

if response is not None:
    html = BeautifulSoup(response.text, 'html.parser')
    paragraphs = html.select("p")
#    for para in paragraphs:
#        print (para.text)

    # just grab the text up to contents as stated in question
    intro = '\n'.join([ para.text for para in paragraphs[0:2]])
    print (intro)


values_for_csv = []
values_for_csv.append(intro)
values_for_csv = pd.DataFrame(values_for_csv)
values_for_csv.to_csv('raw_data.csv', mode = 'a', header = False)    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #url = 'https://en.wikipedia.org/wiki/Linear_algebra'
#content = requests.get(url).content
#soup = BeautifulSoup(content, features = "lxml")

#p_tags = soup.findAll('p')
#a_tags = []
#values_for_csv = list()
#for p_tag in p_tags:
#    a_tags.extend(p_tag.findAll('a'))
#a_tags = [ a_tag for a_tag in a_tags if 'title' in a_tag.attrs and 'href' in a_tag.attrs ]
#for i,a_tag in enumerate(a_tags):
#    print('[{0}] {1}'.format(i+1,a_tag['title']))
#    values_for_csv.append(a_tag['title'])
    
#for names in soup.select('h1[id = "firstHeading"]'):
#    print(names.string)
