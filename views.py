#-*- coding: utf-8 -*-
from django.shortcuts import render
from django import forms
import pickle,numpy,collecteur,sys
# Create your views here.


with open('IndexMotsfile', 'rb') as fp:
    indexMots=pickle.load(fp)
with open('IndexMotsStemfile', 'rb') as fp:
    indexMotsStem=pickle.load(fp)
with open('dictMotsChoisisfile', 'rb') as fp:
    dictMotsChoisis=pickle.load(fp)    
with open('NumPagesAppeleesfile', 'rb') as fp:
    NumPagesAppelees=pickle.load(fp)
with open('pageRanksfile', 'rb') as fp:
    pageRanks=pickle.load(fp)
with open('TitrePageFromNumfile', 'rb') as fp:
    TitrePageFromNum=pickle.load(fp)
print("ok lu!")


def intersect(liste0,liste1):
    if (len(liste0)==0 or len(liste1)==0):
        return []
    i=0
    j=0
    res=[]
    while(i<len(liste0) and j<len(liste1)):
        page0=liste0[i]
        page1=liste1[j]
        if (page0==page1):
            res.append(page0)
            i+=1
            j+=1
        elif (pageRanks[page0]>pageRanks[page1]):
            i+=1
        else:
            j+=1
    return res

def proche(unMot):
    global dictMotsChoisis
    if (unMot in dictMotsChoisis):
        return unMot
    min=sys.maxsize
    for mot in dictMotsChoisis:
        score=collecteur.levenshtein(mot,unMot)
        if (score<min):
            min=score
            plusproche=mot
    if (collecteur.levenshtein(unMot,plusproche)<=1):
        return plusproche
    else:
        return None

def getPagesFromRequete(listemotsdico):
    longueurliste=len(listemotsdico)
    if (longueurliste==0):
        return []
    res=indexMots[listemotsdico[0]]
    indice=1
    while (indice < longueurliste):
        res=intersect(res,indexMots[listemotsdico[indice]])
        indice+=1
    return res

def getPagesFromRequeteStem(listeracinesdico):
    longueurliste=len(listeracinesdico)
    if (longueurliste==0):
        return []
    res=indexMotsStem[listeracinesdico[0]]
    indice=1
    while (indice < longueurliste):
        res=intersect(res,indexMotsStem[listeracinesdico[indice]])
        indice+=1
    return res

def remplacerBlancs(s):
    st=s.replace(" ","_")
    st=st.replace("'","&#39;")
    return st

def rangerPages(listemotsdico,resultatNumPages,n):
    res=resultatNumPages.copy()
    if n>len(listemotsdico):
        return res
    res2=[]
    for p in resultatNumPages:
        compteur=0
        lesmotsdutitre=collecteur.traiter_texte(TitrePageFromNum[p])
        for mot in listemotsdico:
            if mot in lesmotsdutitre:
                compteur+=1
        if compteur>=n:
            if p in res:
                res.remove(p)
                #res.insert(0,p)
                res2.append(p)
                #print("une page deplacée!")
    res2=sorted(res2, key=lambda n:pageRanks[n], reverse=True)
    #resul=res2+res                
    return rangerPages(listemotsdico,res2,n+1)+res 

def rangerPagesRacine(listemotsdico,resultatNumPages,n):
    res=resultatNumPages.copy()
    if n>len(listemotsdico):
        return res
    res2=[]
    for p in resultatNumPages:
        compteur=0
        lesmotsdutitre=collecteur.traiter_texte(TitrePageFromNum[p])
        lesracinesdutitre=[collecteur.racine(m) for m in lesmotsdutitre]
        for mot in listemotsdico:
            if mot in lesracinesdutitre:
                compteur+=1
        if compteur>=n:
            if p in res:
                res.remove(p)
                #res.insert(0,p)
                res2.append(p)
                #print("une page deplacée!")
    res2=sorted(res2, key=lambda n:pageRanks[n], reverse=True)
    #resul=res2+res                
    return rangerPagesRacine(listemotsdico,res2,n+1) + res

"""
def mettreTitresEnTete(listemotsdico,resultatNumPages):
    res=resultatNumPages.copy()
    res2=[]
    for p in resultatNumPages:
        lesmotsdutitre=collecteur.traiter_texte(TitrePageFromNum[p])
        for mot in lesmotsdutitre:
            if mot in listemotsdico:
                if p in res:
                    res.remove(p)
                    #res.insert(0,p)
                    res2.append(p)
                    #print("une page deplacée!")
    res2=sorted(res2, key=lambda n:pageRanks[n], reverse=True)                
    return res2+res
"""

"""
def mettreTitresEnTeteRacine(listemotsdico,resultatNumPages):
    res=resultatNumPages.copy()
    res2=[]
    for p in resultatNumPages:
        lesmotsdutitre=collecteur.traiter_texte(TitrePageFromNum[p])
        lesracinesdutitre=[collecteur.racine(m) for m in lesmotsdutitre]
        for mot in lesracinesdutitre:
            if mot in listemotsdico:
                if p in res:
                    res.remove(p)
                    #res.insert(0,p)
                    res2.append(p)
                    #print("une page deplacée!")
    res2=sorted(res2, key=lambda n:pageRanks[n], reverse=True)
    return res2+res
"""


def traiterRequete(texterequete):
    motsrequete=collecteur.traiter_texte(texterequete)
    listemotsdico=[]
    for mot in motsrequete:
        if mot in indexMots:
            listemotsdico.append(mot)
        else:
            motproche=proche(mot)
            if motproche!=None:
                listemotsdico.append(proche(mot))  
    listemotsdicoaffichage=listemotsdico      
    listemotsdico=sorted(listemotsdico, key=lambda n:len(indexMots[n]))            
    resultatNumPages=getPagesFromRequete(listemotsdico)
    #resultatNumPages=mettreTitresEnTete(listemotsdico,resultatNumPages)
    resultatNumPages=rangerPages(listemotsdico,resultatNumPages,1)
    resultatTitresPages=[TitrePageFromNum[p] for p in resultatNumPages]
    resultatUrlPages=[ "<a href='https://fr.wikipedia.org/wiki/"+remplacerBlancs(t)+"'>"+t+"</a>"   for t in resultatTitresPages]
    requetetraitee=" ".join(listemotsdicoaffichage)
    return requetetraitee,resultatUrlPages

def traiterRequeteRacine(texterequete):
    motsrequete=collecteur.traiter_texte(texterequete)
    listemotsdico=[]
    for mot in motsrequete:
        if mot in indexMots:
            listemotsdico.append(mot)
        else:
            motproche=proche(mot)
            if motproche!=None:
                listemotsdico.append(proche(mot))
    listeracinesdico=[]
    for mot in listemotsdico:
        if collecteur.racine(mot) in indexMotsStem:
            listeracinesdico.append(collecteur.racine(mot))
        # else:
        #     motproche=proche(mot)
        #     if motproche!=None:
        #         listemotsdico.append(proche(mot))
    listeracinesaffichage=listeracinesdico
    listeracinesdico=sorted(listeracinesdico, key=lambda n:len(indexMotsStem[n]))
    resultatNumPages=getPagesFromRequeteStem(listeracinesdico)
    #resultatNumPages=mettreTitresEnTeteRacine(listeracinesdico,resultatNumPages)
    resultatNumPages=rangerPagesRacine(listeracinesdico,resultatNumPages,1)
    resultatTitresPages=[TitrePageFromNum[p] for p in resultatNumPages]
    resultatUrlPages=[ "<a href='https://fr.wikipedia.org/wiki/"+remplacerBlancs(t)+"'>"+t+"</a>"   for t in resultatTitresPages]
    listeracines=[racine+"'" for racine in listeracinesaffichage  ]
    requetetraitee=" ".join(listeracines)
    return requetetraitee,resultatUrlPages

class RequeteForm(forms.Form):
    recherche=forms.CharField(label="votre recherche", max_length=400,widget=forms.TextInput(
                 attrs={'size':'50'}))

def requete(request):
    if request.method=='POST':
        form=RequeteForm(request.POST)
        if form.is_valid():
            texteRequete=form.cleaned_data['recherche']
            #message="coucou"
            #liste=["<a href='https://www.lemonde.fr'>truc</a>", "machin", "bidule"]
            if 'extendedsearch' in request.POST:
                requetetraitee,liste=traiterRequeteRacine(texteRequete)
                message=str(len(liste))+" pages trouvées pour la requete: ("+requetetraitee+")"
                print("recherche élargie!")
            else:
                requetetraitee,liste=traiterRequete(texteRequete)
                message=str(len(liste))+" pages trouvées pour la requete: ("+requetetraitee+")"   
                print("recherche simple")         
    else:
        form=RequeteForm()
    
    return render(request,'requete.html',locals())
