
# coding: utf-8

# ## Exercice 1

# In[1]:


from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer


# In[2]:


import unicodedata,re,string,time,random,pickle,gc,sys


# In[3]:


from bs4 import BeautifulSoup


# In[4]:


def enlever_accents(mot):
    mot_traite = unicodedata.normalize('NFD', mot).encode('ascii', 'ignore')
    mot_traite=mot_traite.decode('utf-8')
    return mot_traite


# In[5]:


tokenizer = RegexpTokenizer('[\w-]+')


# In[6]:


stop_words = list(set(stopwords.words('french')))


# In[7]:


stop_words_sans_accents=list(set([enlever_accents(word) for word in stop_words] + ["http","https","www","les","lui","eux","aux"]))


# In[8]:


#nettoyer tout un texte
def traiter_texte(text):
    letext = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    letext=letext.decode('utf-8')
    letext=letext.lower()  #déjà fait...non!...
    words=tokenizer.tokenize(letext)
    return [word for word in words if ((word not in stop_words_sans_accents) and (len(word)>2))]


# In[9]:


def valide(texte):
    #return (texte.find("technique")>0)
    #return (texte.find("technique")>0 or texte.find("science")>0) 
    return True


# In[10]:


lesPages={}  #pour une page designee par son titre=> (num de page, liste des mots de la page)
lesMots={}  #ensemble des mots de toutes les pages=>nb d'occurences
TitresPagesAppelees={}   #pour une page designée par son titre=>liste des titres des pages appelees
TitrePageFromNum={} #pour une page designee par son num=>son titre
nbpages=0
NumPagesAppelees={} #pour une page désignée par son numero=> liste des numeros de pages appelees


# In[11]:


def get_titres(text):
    regextitre=r'\[\[([\w \-\']+)\|?.*?\]\]'
    return re.findall(regextitre,text)    


# In[12]:


#pour une page constituer la liste des mots et les pages appelées
def traiter_page(compteur,title,text):
    global lesPages,lesMots,TitresPagesAppelees,TitrePageFromNum
    frequMotsPage={}
    lesmotsTitre=traiter_texte(title)
    lesmotsTexte=traiter_texte(text)
    lesmotsTitreEtTexte=lesmotsTitre + lesmotsTexte
    nbMotsPage=len(lesmotsTitreEtTexte)
    for mot in lesmotsTitreEtTexte:
        if mot not in lesMots:
            lesMots[mot]=1
        else:
            lesMots[mot]+=1
        if mot not in frequMotsPage:
            frequMotsPage[mot]=1
        else:
            frequMotsPage[mot]+=1    
    for mot in frequMotsPage:
        frequMotsPage[mot]/=nbMotsPage
    #listeMotsPage=[(mot,frequMotsPage[mot]) for mot in frequMotsPage]
    lesPages[title]=(compteur,frequMotsPage)
    #lesPages[title]=(compteur,list(set(lesmotsTitreEtTexte)))   
    #title=enlever_accents(title).lower()
    TitrePageFromNum[compteur]=title
    TitresPagesAppelees[title]=get_titres(text)   


# In[13]:


#parcourir le fichier et constituer la liste des mots nettoyés pour chaque page
def lire_fichier_xml():
    global nbpages
    start=time.time()
    compteur=0
    numpage=0
    contenu=""
    lire=False
    #lesPages=[]
    #lesPagesMots=[]
    with open( "frwiki-debut.xml","r") as f:
        for line in f:        
            if (compteur==400000):
                break
            pline=line.strip()
            if (len(pline)>=6 and pline[0:6]=="<page>"):
                lire=True
                contenu+=line
            elif (len(pline)>=7 and pline[-7:]=="</page>"):
                lire=False
                contenu+=line
                #contenu.lower() #non pas la peine
                if valide(contenu):
                    soup=BeautifulSoup(contenu)
                    text=soup.find('text').get_text()
                    title=soup.find('title').get_text()
                    traiter_page(compteur,title,text)
                    #lesPages.append([numpage,contenu])
                    #lesPagesMots.append([numpage,traiter_texte(text)])                
                    compteur+=1
                    boucle=compteur%1000
                    if (boucle==0):
                        print("compteur de pages= "+str(compteur)+", nb pages vues= "+str(numpage))
                contenu=""
                numpage+=1                     
            else:
                if (lire):
                    contenu+=line


    nbpages=compteur
    print("total compteur pages= ",compteur)
    print("total pages vues= ",numpage)
    end=time.time()
    print("parcours frwiki: ",end-start)


# In[14]:


gc.collect()


# ## Exercice 2

# In[15]:


import heapq #pour trouver rapidement les 15000 plus fréquents
nbmotschoisis=15000


# In[16]:


#pour trouver la racine des mots
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("french")


# In[17]:


def racine(mot):
    return stemmer.stem(mot)


# In[30]:


#pour corriger les fautes de frappe
def levenshtein(s, t):
        ''' From Wikipedia article; Iterative with two matrix rows. '''
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]


# In[31]:


def proche(unMot):
    global dictMotsChoisis
    if (unMot in dictMotsChoisis):
        return unMot
    min=sys.maxsize
    for mot in dictMotsChoisis:
        if (levenshtein(mot,unMot)<min):
            min=levenshtein(mot,unMot)
            plusproche=mot
    if (levenshtein(unMot,plusproche)<=2):
        return plusproche
    else:
        return None


# ## exercice 3 et  4

# In[32]:


import numpy as np


# In[33]:


class Matrice:
    def __init__(self,C=[],L=[],I=[]):
        self.C=C
        self.L=L
        self.I=I
        
    def construire_from_matrice2D(self,M):
        n=M.shape[0]
        m=len([M[i][j] for i in range(n) for j in range(n) if (M[i][j]!=0)])
        indice=0
        self.C=[0.0 for i in range(m)]
        self.I=[0 for i in range(m)]
        self.L=[0 for i in range(n+1)]
        self.L[n]=m
        fait=[False for i in range(n)]
        for i in range(n):
            lignezeros=True
            for j in range(n):
                if (M[i][j]!=0):
                    lignezeros=False
                    self.C[indice]=M[i][j]
                    if (not fait[i]):
                        self.L[i]=indice
                        fait[i]=True
                    self.I[indice]=j
                    indice+=1
            #if (lignezeros):
            #    self.L[i]=self.L[i+1]
        for i in range(n-1,-1,-1):
            lignezeros=True
            for j in range(n):
                if (M[i][j]!=0):
                    lignezeros=False
            if (lignezeros):
                self.L[i]=self.L[i+1]

                     
    def construire_matrice2D(self,C,L,I):
        n=len(L)-1
        self.C=C
        self.L=L
        self.I=I
        m=len(C)
        M=np.zeros((n,n))
        for i in range(n):
            for c in range(L[i],L[i+1]):
                M[i][I[c]]=C[c]
        return M
            
                
    def produitV(self,V):
        dimV=len(V)
        n=len(self.L)-1
        if (dimV!=n):
            print("erreur dans les dimensions")
            return None    
        P=np.zeros(n)
        for i in range(n):
            for c in range(self.L[i],self.L[i+1]):
                P[i]+=self.C[c]*V[self.I[c]]
        return P
            
                
    def print(self):
        print("C:")
        print(self.C)
        print("L:")
        print(self.L)
        print("I:")
        print(self.I)
        


# ## Exercice 5

# ## Exercice 6

# ### constitution de l'index Mots->Pages

# In[34]:


gc.collect()


# In[35]:


#constitution de l'index des Mots indexMots: mot=>liste des numeros de pages contenant le mot
indexMots={}
def constituer_index_mots():
    global indexMots,lesPages,dictMotsChoisis
    start=time.time()
    seuilfreq=0.0000  #seuil de frequence des mots sur une page
    compt=0
    for page in lesPages:
        numpage=lesPages[page][0]
        frequMotsPage=lesPages[page][1]
        for (mot,freq) in frequMotsPage.items():
            if mot in dictMotsChoisis:
                if freq>seuilfreq:
                    if mot not in indexMots:
                        indexMots[mot]=[numpage]
                    else: #(numpage not in indexMots[mot]):
                        indexMots[mot].append(numpage)
        compt+=1
        if (compt%1000==0):
            print("indexage pages: ", compt)
    end=time.time()
    print("temps pour index = ",end-start)


# In[36]:


gc.collect()


# In[37]:


#avec stemming
#constitution de l'index des racines indexMotsStem: racine=>liste des numeros de pages contenant un mot ayant cette racine
indexMotsStem={}
def constituer_index_racines():
    global indexMotsStem,indexMots
    start=time.time()
    for mot in indexMots:
        motStem=racine(mot)
        if motStem not in indexMotsStem:
            indexMotsStem[motStem]=indexMots[mot]
        else:
            set1=set(indexMotsStem[motStem])
            set2=set(indexMots[mot])
            setdelta=set2-set1
            #indexMotsStem[motStem]+=indexMots[mot]  #TODO eliminer les doubles!!
            indexMotsStem[motStem]+=list(setdelta)  #fait!!!
    end=time.time()
    print("temps pour index Stem = ",end-start)


# ### graphe

# In[38]:


gc.collect()


# In[39]:


#construction du graphe NumPagesAppelees: numpage=>liste des numeros des pages appelees
def constituer_graphe():
    global NumPagesAppelees,TitresPagesAppelees,lesPages
    NumPagesAppelees={} 
    start=time.time()
    compt=0
    for titrepage in TitresPagesAppelees:
        for titreappele in TitresPagesAppelees[titrepage]:
            if titreappele in TitresPagesAppelees:
                num=lesPages[titrepage][0]
                numappele=lesPages[titreappele][0]
                if num not in NumPagesAppelees:
                    NumPagesAppelees[num]=[numappele]
                elif (numappele not in NumPagesAppelees[num]):
                    NumPagesAppelees[num].append(numappele)
        compt+=1
        if (compt%1000==0):
            print("graphe pages: ",compt)
    end=time.time()
    print("construction du graphe: ", end-start)


# In[42]:


#constitution de C,L,I 
def CLIfromGraphe(nbpages,NumPagesAppelees):
    C=[]
    L=[]
    I=[]
    indexC=0
    for numpage in range(nbpages):
        if numpage in NumPagesAppelees:
            longu=len(NumPagesAppelees[numpage])
            for i in range(longu):
                C.append(1/longu)
            L.append(indexC)
            indexC+=longu
            for numpageappelee in NumPagesAppelees[numpage]:
                I.append(numpageappelee) 
        else:
            L.append(0)
    L.append(len(C))
    for numpage in range(nbpages-1,-1,-1):
        if numpage not in NumPagesAppelees:
            L[numpage]=L[numpage+1]   
    return C,L,I


# ## TP2

# ### exercice 1

# In[43]:


#multiplication par la matrice transposée donnée par C,L,I
def produitVT(C,L,I,V):
    n=len(L)-1
    m=len(C)
    dimV=len(V)
    n=len(L)-1
    if (dimV!=n):
        print("erreur dans les dimensions")
        return None    
    P=np.zeros(n)
         
    for i in range(n):
        for c in range(L[i],L[i+1]):
            P[I[c]]+=C[c]*V[i]
    return P       


# In[44]:


#multiplication par la matrice transposée donnée par C,L,I en traitant les lignes nulles
def produitVTlignes0(C,L,I,V):
    n=len(L)-1
    m=len(C)
    dimV=len(V)
    n=len(L)-1
    if (dimV!=n):
        print("erreur dans les dimensions")
        return None    
    P=np.zeros(n)
    nbaleas=n//300
    if (nbaleas==0):
        nbaleas=1
    delta=1/nbaleas
    for i in range(n):
        if (L[i]==L[i+1]):
            aleas=random.sample(range(0, n), nbaleas)
            for j in aleas:
                P[j]+=delta*V[i]
        else:
            for c in range(L[i],L[i+1]):
                P[I[c]]+=C[c]*V[i]
    return P 


# In[46]:


#exemple simple
tab=Matrice()
M=np.array([[0,1,0,2],[1,0,0,0],[0,0,0,0],[2,3,4,0]])
tab.construire_from_matrice2D(M)


# In[47]:


V=[1,2,3,4]


# In[49]:


tp=tab.produitV(V)


# In[50]:


pv=produitVT(tab.C,tab.L,tab.I,V)


# In[51]:


#autre exemple simple
M=np.array([[1,4,0,5],[1,0,0,1],[0,1,0,2],[1,0,0,1]])
tab.construire_from_matrice2D(M)
V=[1,2,1,4]


# In[52]:


ar=tab.produitV(V)


# In[53]:


pv=produitVT(tab.C,tab.L,tab.I,V)


# ### exercice 2

# In[54]:


convergence=[]


# In[55]:


#page rank avec un nombre de pas
def pageRankPas(C,L,I,Z,pas):
    global convergence
    convergence=[]
    P1=Z
    compt=0
    for i in range(pas):
        P2=produitVTlignes0(C,L,I,P1)
        #print("P=",P)
        diff=norm1(P1,P2)
        convergence.append(diff)
        P1=P2
        compt+=1
        if (compt%10==0):
            print("boucle:",compt)
            print("somme=",sum(P2))
    return P1    


# In[56]:


def norm1(P1,P2):
    if (len(P1)!=len(P2)):
        print("dans norm1 erreur sur les dimensions")
    n=len(P1)
    res=sum([abs(P1[i]-P2[i]) for i in range(n)])
    #print("res=",res)
    return res


# In[57]:


#page rank avec epsilon
def pageRankEpsilon(C,L,I,Z,epsilon):
    global convergence
    convergence=[]
    P1=Z
    diff=sys.maxsize
    compt=0
    while (diff>epsilon):
        P2=produitVTlignes0(C,L,I,P1)
        #print("P2=",P2)
        diff=norm1(P1,P2)
        convergence.append(diff)
        P1=P2  
        compt+=1
        if (compt%50==0):
            print("boucle:",compt)
            print("somme=",sum(P2))
            print("diff=",diff)
    return P2    


# In[60]:


#page rank avec un nombre de pas
def pageRankZapPas(C,L,I,Z,d,pas):
    global convergence
    convergence=[]
    n=len(L)-1
    P1=Z
    compt=0
    for i in range(pas):
        P2=d/n + (1-d)*produitVTlignes0(C,L,I,P1)
        #print("P=",P)
        diff=norm1(P1,P2)
        convergence.append(diff)
        P1=P2
        compt+=1
        if (compt%10==0):
            print("boucle:",compt)
            print("somme=",sum(P2))
    return P2    


# In[61]:


#page rank avec epsilon
def pageRankZapEpsilon(C,L,I,Z,d,epsilon):
    global convergence
    convergence=[]
    n=len(L)-1
    P1=Z
    diff=sys.maxsize
    compt=0
    while (diff>epsilon):
        P2=d/n + (1-d)*produitVTlignes0(C,L,I,P1)
        #print("P2=",P2)
        diff=norm1(P1,P2)
        convergence.append(diff)
        P1=P2     
        compt+=1
        if (compt%50==0):
            print("boucle:",compt)
            print("somme=",sum(P2))
            print("diff=",diff)
    return P2    


# In[62]:


d=0.15


# In[63]:


e=2e-3


# In[64]:


import matplotlib.pyplot as plt


# In[65]:


#plt.plot(convergence)


# In[66]:


gc.collect()


# # ici commence l'execution

# In[67]:


if __name__=='__main__':
    lire_fichier_xml()
    MotsChoisisTuples = heapq.nlargest(nbmotschoisis, lesMots.items(), key=lambda kv: kv[1])
    MotsChoisis=[MotsChoisisTuples[i][0] for i in range(nbmotschoisis)]
    dictMotsChoisis={}
    for mot in MotsChoisis:
        dictMotsChoisis[mot]=True
    constituer_index_mots()
    constituer_index_racines()
    constituer_graphe()
    C,L,I=CLIfromGraphe(nbpages,NumPagesAppelees)
    p0=1/nbpages
    Z=[p0 for i in range(nbpages)]
    convergence=[]
    pageRanks=pageRankZapPas(C,L,I,Z,d,60)
    for mot in indexMots:
        indexMots[mot]=sorted(indexMots[mot], key=lambda n:pageRanks[n], reverse=True) 
    for mot in indexMotsStem:
        indexMotsStem[mot]=sorted(indexMotsStem[mot], key=lambda n:pageRanks[n], reverse=True)


# In[68]:


if __name__=='__main__':
    start=time.time()
    with open('IndexMotsfile-debut', 'wb') as fp:
         pickle.dump(indexMots, fp)
    with open('IndexMotsStemfile-debut', 'wb') as fp:
         pickle.dump(indexMotsStem, fp)
    with open('dictMotsChoisisfile-debut', 'wb') as fp:
         pickle.dump(dictMotsChoisis, fp)    
    with open('NumPagesAppeleesfile-debut', 'wb') as fp:
         pickle.dump(NumPagesAppelees, fp)
    with open('pageRanksfile-debut', 'wb') as fp:
         pickle.dump(pageRanks, fp)
    with open('TitrePageFromNumfile-debut', 'wb') as fp:
         pickle.dump(TitrePageFromNum, fp)
    end=time.time()
    print("sauvegarde CLI:",end-start)

