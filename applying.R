#Copyright (c) <2020>, <postaxmi@gmail.com>
#  All rights reserved.

#This source code is licensed under the GPL-style license found in the
#LICENSE file in the root directory of this source tree. 

library(tau)# for textcnt
library(caret)#for data partitioning
library(openNLP)#for POS en,nl
require(NLP)#for openNLP
library(koRpus)#for POS es
library(rJava)#for POS gr
library(rjson)#for loading content.json (contains info about problems)
library(rpart)
library(e1071)
library(randomForest)
#encoding
Sys.setlocale(category="LC_ALL",locale="it_IT.UTF-8")
#trasforma il testo in simboli
getTextShape<-function(text){
  result<-trim(text)
  result<-gsub("[[:lower:]]+","l",result)#parola tutta in minuscolo
  result<-gsub("[[:alpha:]]*[[:upper:]][[:alpha:]]*[[:upper:]][[:alpha:]]*","w",result)#parola con 2 o
  pi?? maiuscole
  result<-gsub("[[:upper:]][[:lower:]]*","u",result)#parola con una maiuscola all'inizio (o solo lettera  maiuscola)
# result<-gsub("(?<![[:lower:]])[[:upper:]](?![[:alpha:]])","A",result,perl=TRUE)#una singola lettera maiuscola
result<-gsub("[[:digit:]]+","N",result)#numero
return(result)
}
#word lower and upper case, number, punctution
getNgramsShape<-function(text,N){
  shape<-getTextShape(text)
  shape<-gsub("_","s",shape)#problemi con textcnt e l'underscore quindi sostituito con s
  res<-textcnt(shape,split="[[:space:]]+",method="ngram",n=N)
  tmp<-list()
  for(i in names(res)){#remove ngram with blank character (put by textcnt at beginning and end)
    if(!(grepl("_",i))){
      tmp[[i]]<-res[[i]]
    }
  }
  names(tmp)<-paste0("ts#",names(tmp))
  return (unlist(tmp))
}
#restituisce distanza
getIntervalDistance<-function(text){
  shape<-getTextShape(text)
  arr<-unlist(strsplit(shape,""))
  all<-unique(arr)
  result<-data.frame(Symbol=all,Mean=numeric(length(all)),Dev=numeric(length(all)),Occurrences
                     =numeric(length(all)))
  for(s in all){
    positions<-which(s==arr)
    result[result$Symbol==s,"Occurrences"]<-length(positions)
    if(length(positions)>1){
      d<-diff(positions)
      result[result$Symbol==s,"Mean"]<-mean(d)
      result[result$Symbol==s,"Dev"]<-sd(d)
    }
    else{
      result[result$Symbol==s,"Mean"]<-NA
      result[result$Symbol==s,"Dev"]<-NA
    }
  }
  return(result)
}
#restituisce per ciascun simbolo (vedi text shape) le posizioni corrispondenti nel testo
getIndexesSym<-function(text){
  shape<-getTextShape(text)
  arr<-unlist(strsplit(shape,""))
  all<-unique(arr)
  result<-list()
  for(a in all){
    result[[a]]<-list()
    result[[a]]<-which(a==arr)
  }
  return(result)
}
#return POS tags of the text, for english and dutch
getPOS<-function(text, sent_token_annotator,word_token_annotator,pos_tag_annotator){
  s<-as.String(text)
  s2 <- NLP::annotate(s, list(sent_token_annotator, word_token_annotator))
  s3 <- NLP::annotate(s, pos_tag_annotator, s2)
  s3w <- subset(s3, type == "word")
  return (sapply(s3w$features, '[[' , "POS"))
}
#input file path, output POS tags, for spanish
getPOSES<-function(file){
  return ((treetag(file,treetagger="~/treetagger/cmd/tree-tagger-spanish",
                   lang="es"))@TT.res$tag)
}
#for POS gr from Java
.jinit(".",force.init=TRUE)
.jaddClassPath(dir("/home/postaxmi",full.names=TRUE))#path of *.class files
getPOSGRobj<-.jnew("getPOSGR")
#absolute path file
getPOSGR<-function(file){
  result<-.jcall(getPOSGRobj,"[S","getTags",file)
  return(result)
}
#cavnar trenkle
#get vector with rank positions
getRanking<-function(a){
  cnt <- sort(a,decreasing=TRUE)
  cntnames <- names(cnt)
  cnt <- 1:length(cnt)#rank position
  names(cnt) <- cntnames
  return(cnt)
}
#returns out-of-rank score for the two vecors (same length)
ct <- function(aa,bb) {
  primo<-getRanking(aa)
  secondo<-getRanking(bb)
  x <- abs(secondo[names(primo)]-primo[names(primo)])
  # print(x)
  y <- abs(primo[names(secondo)]-secondo[names(secondo)])
  # print(y)
  #return(x)
  return(sum(x))
}
#"normalize vector": vector with name=value vector and vector with names names,
#return vector with all names of N and value from v, if name not present value 0
normalize<-function(vector,names){
  result<-c()
  nvector<-names(vector)
  common<-intersect(names,nvector)
  result[common]<-vector[common]
  result[setdiff(names,nvector)]<-0
  return(result)
}
distCT<-function(a,b){
  #allNames<-union(names(a),names(b))
  aa<-normalize(a,union(names(a),names(b)))
  bb<-normalize(b,union(names(a),names(b)))
  res<-ct(aa,bb)
  # print(res)
  return(res)
}
getThreshold<-function(distances,method){
  ms<-numeric(2)
  if(method=="mean"){
    ms[[2]]<-mean(distances[distances$GrandeVeritas==0,1])
    ms[[1]]<-mean(distances[distances$GrandeVeritas==1,1])
  }else if(method=="median"){
    ms[[2]]<-median(distances[distances$GrandeVeritas==0,1])
    ms[[1]]<-median(distances[distances$GrandeVeritas==1,1])
  }else if(method=="best"){
    ms[[2]]<-median(distances[distances$GrandeVeritas==0,1])
    ms[[1]]<-median(distances[distances$GrandeVeritas==1,1])
    howMany<-10
    thres<-seq.int(ms[[1]],ms[[2]],length.out=howMany)
    acc<-numeric(howMany)
    for(i in 1:howMany){
      acc[[i]]<-getAccuracy(thres[[i]],distances)
    }
    print(thres)
    print(acc)
    return(thres[[which.max(acc)]])#return threshold with maximum accuracy, improve taking the max in the middle not the first one
  }
  return(mean(ms))
}
getAccuracy<-function(threshold,instances){
  correct<-nrow(instances[instances$GrandeVeritas==1 &
                            instances[,1]<threshold,])+nrow(instances[instances$GrandeVeritas==0 &
                                                                        instances[,1]>=threshold,])
  return(correct/nrow(instances))
}
#per ogni parola in words restiuisce la lunghezza(in caratteri)
getLengthsWord<-function(words){
  result<-sapply(words,nchar)#lunghezza in base a numero caratteri
  return (result)
}
#restituisce tabella(quante parole di lunghezza n caratteri per ogni n preesente nelle parole)
getWordLengthCount<-function(words){
  lengths<-getLengthsWord(words)
  result<-table(lengths)
  names(result)<-paste0("Wl#",names(result))
  return (result)
}
#restituisce tabella(quante parole di lunghezza n caratteri per ogni n preesente nelle parole)
getAllWordLengthCount<-function(text){
  words<-unlist(strsplit(text,split="[[:space:][:punct:][:digit:]]+"))
  words<-words[words!=""]
  lengths<-getLengthsWord(words)
  result<-table(lengths)
  names(result)<-paste0("Wl#",names(result))
  return (result)
}
#numero di parole diverse in rapporto alle parole totali
#input lista con nomi: per ogni nome valore delle occorenze
getWordsRichness<-function(words){
  return (list("Wr"=length(words)/sum(words)))
}
#remove multi empty spaces
trim <- function(x) return(gsub("^ *|(?<= ) | *$", "", x, perl=T))
#per ogni frase del testo restiuisce la lunghezza(in token)
getLengthsSentence<-function(text){
  sentences<-unlist(strsplit(trim(text),split="[.?!;:]+"))
  # result[[1]]<-lengthsChar<-sapply(sentences,nchar)#lunghezza in base a numero caratteri
  result<-sapply(sentences,function(x) length(strsplit(x,' ')[[1]]))#lunghezza in base a token (separati da spazio)
  return (result)
}
#restituisce tabella(quante frasi di lunghezza n token per ogni n preesente nel testo)
getSentenceLengthCount<-function(text){
  lengths<-getLengthsSentence(text)
  result<-table(lengths)
  names(result)<-paste0("Sl#",names(result))
  return (result)
}
#input testo, output al posto di ogni frase il gruppo di appartenenza della frase
getSentenceLengthGroups<-function(text, sentenceGroups){
  lengths<-unlist(getLengthsSentence(text))
  result<-sapply(lengths,function(x,y) getGroup(y,x),y=sentenceGroups)
  return (result)
}
#groups: dataframe with pairs <group,value>, sort in ascending order by value
#compare input value with each pair.value, return the group with the nearest pair.value<=value
getGroup<-function(groups,value){
  for(r in 1:nrow(groups)){
    if(value<=groups[r,"Value"]){
      return(groups[r,"Group"])
    }
  }
  return (groups[nrow(groups),"Group"])
}
#restituisce solo la punteggiatura presente nel testo ricevuto come input
getPunctuation<-function(text){
  return(gsub("[^.,;:!?\"]", "", text, perl=T))
}
getPunctNgrams<-function(text,N){
  ps<-getPunctuation(text)#remove all characters that are not punctuation
  res<-textcnt(ps,split="[[:space:]]+",method="ngram",n=N)#ngrams
  tmp<-list()
  for(i in names(res)){#remove ngram with blank character (put by textcnt at beginning and end)
    if(!(grepl("_",i))){
      tmp[[i]]<-res[[i]]
    }
  }
  names(tmp)<-paste0("p#",names(tmp))
  return (tmp)
}
getCharNgrams<-function(text,N){
  res<-textcnt(text,n=N,method="ngram",decreasing=TRUE,tolower=FALSE)
  names(res)<-paste0("C#",names(res))
  return(res)
}
getWordNgrams<-function(text,N){
  res<-textcnt(text,n=N,method="string")
  names(res)<-paste0("W#",names(res))
  return(res)
}
getPOSNgrams<-function(text,N){
  res<-textcnt(text,split=" ",n=N,method="string")
  names(res)<-paste0("P#",names(res))
  return(res)
}
getSentGroupNgrams<-function(text,N){
  res<-textcnt(text,split="ngram",n=2L)
  names(res)<-paste0("S#",names(res))
  return(res)
}
#responses sono le risposte date, solutions sono le risposte corrette, restituisce l'accuracy
evaluateAccuracy<-function(responses,solutions){
  positive<-responses>0.5
  return((length(which(positive&solutions==1))+length(which(!positive&solutions==0)))/length(responses))
}
#param = commandArgs()[-(1:3)];
#inputPath<-param[[1]]#"~/pan15EN/"
#outputPath<-param[[2]]#"~/"
#inputPath<-"/media/training-datasets/pan15-authorship-verification-training-dataset-english-2015-03-02/"
inputPath<-"/home/postaxmi/pan2/panSP/"
outputPath<-"/home/postaxmi/"
infoInput <- fromJSON(file =paste0(inputPath,"contents.json"))
languageName<-infoInput[["language"]]
problemNames<-infoInput[["problems"]]
if(languageName=="English"){
language="EN"
}else if(languageName=="Dutch"){
language="DU"
}else if(languageName=="Spanish"){
language="SP"
}else if(languageName=="Greek"){
language="GR"
}
problemDirs<-problemNames
path<-inputPath
allDocs<-list() #global list with all documents in the corpus
allDocsPOSed<-list()#global list with all documents transformed to POS in the corpus
allDocsSl<-list()#global list with sentence length of all documents
start<-proc.time()
minWordNgrams<-1
maxWordNgrams<-3
wordNgrams<-readLines(paste("/home/postaxmi/W",language,500,sep=""))
maxCharNgrams<-5
charNgrams<-readLines(paste("/home/postaxmi/C",language,500,sep=""))
minPOSNgrams<-1
maxPOSNgrams<-3
POSNgrams<-readLines(paste("/home/postaxmi/P",language,500,sep=""))
maxPunctuationNgrams<-2
punctuationNgramsTot<-readLines(paste("/home/postaxmi/p",language,sep=""))
maxTextShapeNgrams<-3
textShapeNgramsTot<-readLines(paste("/home/postaxmi/ts",language,100,sep=""))
sentenceGroups<-read.table(paste("/home/postaxmi/Sg",language,sep=""))
combResp<-"Mean"
ngramsSl<-list()
for(i in 1:ng){
ngramsSl[[length(ngramsSl)+1]]<-paste0("S#",i)
for(j in 1:ng){
  ngramsSl[[length(ngramsSl)+1]]<-paste0("S#",i,j)
}
}
ngramsSl<-unlist(ngramsSl)
models<-list()
for(f in 1:9){
  models[[f]]<-readRDS(file=paste0("rf",f,language))
}
print(proc.time()-start)
#read all documents
start<-proc.time()
if(language=="EN"||language=="DU"){
  #annotators for POS, english and dutch
  languagePOS<-ifelse(language=="EN","en","nl")
  sent_token_annotator <- Maxent_Sent_Token_Annotator(language=languagePOS)
  word_token_annotator <- Maxent_Word_Token_Annotator(language=languagePOS)
  pos_tag_annotator <- Maxent_POS_Tag_Annotator(language=languagePOS)
  for(problem in problemDirs){ #for all problems
    print(problem)
    documents<-list.files(paste(path,problem,sep=""))#documents for this problem
    for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,
                                                                        sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOS(allDocs[[paste(problem,document,sep='-')]],sent_token_annotator,word_token_annotator,pos_tag_annotator),collapse=" ")
      allDocsSl[[paste(problem,document,sep='-')]]<-paste(getSentenceLengthGroups(allDocs[[paste(
        problem,document,sep='-')]],sentenceGroups),collapse="")
    }
  }
}else if(language=="SP"){
  for(problem in problemDirs){ #for all problems
    print(problem)
    documents<-list.files(paste(path,problem,sep=""))#documents for this problem
    for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,
                                                                        sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOSES(paste(path,problem,"/",document,sep="")),collapse=" ")
      allDocsSl[[paste(problem,document,sep='-')]]<-paste(getSentenceLengthGroups(allDocs[[paste(
        problem,document,sep='-')]],sentenceGroups),collapse="")
    }
  }
}else if(language=="GR"){
  for(problem in problemDirs){ #for all problems
    print(problem)
    documents<-list.files(paste(path,problem,sep=""))#documents for this problem
    for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,
                                                                        sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOSGR(paste(path,problem,"/",document,sep="")),collapse=" ")
      allDocsSl[[paste(problem,document,sep='-')]]<-paste(getSentenceLengthGroups(allDocs[[paste(
        problem,document,sep='-')]],sentenceGroups),collapse="")
    }
  }
}
print(proc.time()-start)#470s 100 EN problems
#for each problem get occurences for all ngram retrieved previously
start<-proc.time()
problems<-list()
#lunghezza di ogni documento
lengthDoc<-data.frame(Problema=character(0),NomeDoc=character(0),TotalLength=numeric(0)
)
columnsTmp<-list()
#per lunghezza frasi
sentencesLength<-data.frame(stringsAsFactors=FALSE)#conterr?? numero totale(sull'intero dataset) di frasi con lunghezza n, n=1,2,3...
sentencesLength[1,]<-0
sentencesLength[,paste0("Sl#",as.character(1:120))]<-0
totDocs<-0
for(problem in problemDirs){ #for all problems
  print(problem)
  documents<-list.files(paste(path,problem,sep=""))#documents for this problem
  #it will contains 3 dataframes, (1) for word ngram, (2) for char ngram and (3) for pos ngram
  problems[[problem]]<-lapply(1:9,function(x,y)
    data.frame(NomeDoc=y,stringsAsFactors=FALSE),y=documents)
  problems[[problem]][[1]][,wordNgrams]<-0
  problems[[problem]][[2]][,charNgrams]<-0#add as columns all char ngrams retrieved
  #add as columns all POS ngrams retrieved
  problems[[problem]][[3]][,POSNgrams]<-0
  problems[[problem]][[4]][,paste0("Wl#",as.character(1:50))]<-0#per lunghezza parole
  problems[[problem]][[5]][,paste0("Sl#",as.character(1:120))]<-0#per lunghezza sentences
  problems[[problem]][[6]][,"Wr"]<-0#per word richness
  problems[[problem]][[7]][,punctuationNgramsTot]<-0#per punctuation ngrams
  problems[[problem]][[8]][,textShapeNgramsTot]<-0#per textshape ngrams
  problems[[problem]][[9]][,ngramsSl]<-0#per sentences groups
  for(document in documents){#for all documents of this problem
    #word ngrams
    wordNgramsDoc<-unlist(lapply(minWordNgrams:maxWordNgrams,function(x,y)
      getWordNgrams(y,x),y=allDocs[[paste(problem,document,sep="-")]]))
    columnsTmp<-intersect(names(wordNgramsDoc),wordNgrams)
    problems[[problem]][[1]][problems[[problem]][[1]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=wordNgramsDoc)
    #lengthDoc<-rbind(lengthDoc,data.frame(Problema=problem,NomeDoc=document,TotalLength=sum(problems[[problem]][[1]][problems[[problem]][[1]]$NomeDoc==document,2:ncol(problems[[problem]][[1]])]),stringsAsFactors=FALSE))
#char ngrams
charNgramsDoc<-getCharNgrams(allDocs[[paste(problem,document,sep="-")]],maxCharNgrams)#textcnt(allDocs[[paste(problem,document,sep="-")]],method="ngram",n=maxCharNgrams,tolower=FALSE)#char ngrams for this document
columnsTmp<-intersect(names(charNgramsDoc),charNgrams)
problems[[problem]][[2]][problems[[problem]][[2]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=charNgramsDoc)
#POS ngrams
POSNgramsDoc<-unlist(lapply(minPOSNgrams:maxPOSNgrams,function(x,y)
  getPOSNgrams(y,x),y=allDocsPOSed[[paste(problem,document,sep="-")]]))
columnsTmp<-intersect(names(POSNgramsDoc),POSNgrams)
problems[[problem]][[3]][problems[[problem]][[3]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=POSNgramsDoc)
#lengths of words and sentences
#lengths<-getWordLengthCount(names(textcnt(allDocs[[paste(problem,document,sep="-")]],method="string",n=1L)))#lunghezza parole contando sulle parole uniche
lengths<-getAllWordLengthCount(allDocs[[paste(problem,document,sep="-")]])#lunghezza parole contando su tutte le parole
columnsTmp<-intersect(names(lengths),colnames(problems[[problem]][[4]]))
problems[[problem]][[4]][problems[[problem]][[4]]$NomeDoc==document,names(lengths)]<-lengt
hs
lengths<-getSentenceLengthCount(allDocs[[paste(problem,document,sep="-")]])
columnsTmp<-intersect(names(lengths),colnames(problems[[problem]][[5]]))
problems[[problem]][[5]][problems[[problem]][[5]]$NomeDoc==document,names(lengths)]<-lengt
hs
#richness of vocabulary
wR<-getWordsRichness(textcnt(allDocs[[paste(problem,document,sep="-")]],method="string",n=
                               1L))
problems[[problem]][[6]][problems[[problem]][[6]]$NomeDoc==document,names(wR)]<-wR
#punctuation ngrams
punctuationNgramsDoc<-getPunctNgrams(allDocs[[paste(problem,document,sep="-")]],maxPunctuationNgrams)
columnsTmp<-intersect(names(punctuationNgramsDoc),punctuationNgramsTot)
problems[[problem]][[7]][problems[[problem]][[7]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=punctuationNgramsDoc)
#textshapeNgrams
textShapeNgramsDoc<-getNgramsShape(allDocs[[paste(problem,document,sep="-")]],maxTextShapeNgrams)
columnsTmp<-intersect(names(textShapeNgramsDoc),textShapeNgramsTot)
problems[[problem]][[8]][problems[[problem]][[8]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=textShapeNgramsDoc)
#Sg
ngramsSlDoc<-getSentGroupNgrams(allDocsSl[[paste(problem,document,sep='-')]],2)#textcnt(allDocsSl[[paste(problem,document,sep='-')]],split="ngram",n=2L)
columnsTmp<-intersect(names(ngramsSlDoc),ngramsSl)
problems[[problem]][[9]][problems[[problem]][[9]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=ngramsSlDoc)
  }
  for(l in colnames(sentencesLength)){
    sentencesLength[[l]]<-sentencesLength[[l]]+sum(problems[[problem]][[5]][,l])
  }
  totDocs<-totDocs+nrow(problems[[problem]][[5]])
  #relative frequencies
  for(i in 1:9){#for all type of ngrams
    if(i!=6){#no frequency for word richness, there's only a value
      nColumns<-ncol(problems[[problem]][[i]])
      for(r in 1:nrow(problems[[problem]][[i]])){#for all documents (rows)
        problems[[problem]][[i]][r,2:nColumns]<-prop.table(problems[[problem]][[i]][r,2:nColumns])
      }
    }
  }
}
print(proc.time()-start) #(234s 100 EN problems all unigrams)
#allInstFeats<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)
distinctInstFeats<-list()
for(i in 1:9){
  #allInstFeats[,colnames(problems[[1]][[i]])[-1]]<-0
  distinctInstFeats[[i]]<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)
  distinctInstFeats[[i]][,colnames(problems[[1]][[i]])[-1]]<-0
}
start<-proc.time()
#put the values for each problem
for(problem in problemDirs){
  print(problem)
  unk<-list()#it will contain the 3 dataframes for the unknown document (only a row)
  kno<-list()#it will contain the 3 dataframes for the known documents (only a row, average of known docs)
for(i in 1:9){#for all type of ngrams (1 word,2 char, 3 POS) and lengths (4 word, 5 sentences)
  #kno[[i]]<-sapply(2:ncol(problems[[problem]][[i]]),function(x,y,z) weighted.mean(y[,x],z),y=problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc!="unknown.txt",],z=lengthDoc[lengthDoc$Problema==problem & lengthDoc$NomeDoc!="unknown.txt","TotalLength"])
#distancese between known and unknown
if(i==6){
  unk[[i]]<-problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc=="unknown.txt",2:ncol(problems[[problem]][[i]])]
  #aggregate all known document with weighted mean (the weight is the length of the document)
kno[[i]]<-mean(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc!="unknown.txt",2:ncol(
  problems[[problem]][[i]])])
distinctInstFeats[[i]][distinctInstFeats[[i]]$Problema==problem,"Wr"]<-abs(kno[[i]]-unk[[i]])
# allInstFeats[allInstFeats$Problema==problem,"Wr"]<-abs(kno[[i]]-unk[[i]])
}else{
  unk[[i]]<-unlist(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc=="unknown.txt",2:ncol
                                            (problems[[problem]][[i]])])
  #aggregate all known document with weighted mean (the weight is the length of the document)
kno[[i]]<-unlist(sapply(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc!="unknown.txt",
                                                 2:ncol(problems[[problem]][[i]])],mean))
distinctInstFeats[[i]][distinctInstFeats[[i]]$Problema==problem,names(kno[[i]])]<-abs(kno[[i]]-unk[[
  i]])
# allInstFeats[allInstFeats$Problema==problem,names(kno[[i]])]<-abs(kno[[i]]-unk[[i]])
}
}
}
print(proc.time()-start)#(140s for 100 EN problems all unigrams)
proc.time()
modelNames<-c("RandF")
modelFeatNames<-list()
for(m in modelNames){
  for(f in 1:(length(distinctInstFeats))){
    modelFeatNames[length(modelFeatNames)+1]<-paste0(m,f)
  }
}
modelFeatNames<-unlist(modelFeatNames)
responses<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)#contiene risposte date dai vari modelli (uno per colonna) per ogni problema(uno per riga)
responsesComb<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)#contiene risposte aggregate dei modelli dello stesso tipo (aggragazione con mean o majority)
testSetMod<-list()
for(f in 1:length(distinctInstFeats)){
  testSetMod[[f]]<-distinctInstFeats[[f]]
  colnames(testSetMod[[f]])[2:(ncol(testSetMod[[f]]))]<-paste0("F",seq(1:(ncol(testSetMod[[f]])-1)))
  #togliere colonne
  if(f==4){
    testSetMod[[f]]<-testSetMod[[f]][,c(1:16,ncol(testSetMod[[f]]))]
  }
  if(f==5){
    testSetMod[[f]]<-testSetMod[[f]][,c(1:40,ncol(testSetMod[[f]]))]
  }
  predictions<-list()
  for(i in 1:length(modelNames)){
    predictions[[i]]<-predict(models[[f]],testSetMod[[f]])
    responses[,paste0(modelNames[[i]],f)]<-predictions[[i]]
  }
}
#ora responses contiene una riga per ciascun problema e su ogni colonna la risposta data da un certo modello basato su una certo tipo di feature
for(m in modelNames){
  respCols<-list()
  for(f in 1:length(distinctInstFeats)){
    respCols[length(respCols)+1]<-paste0(m,f)
  }
  respCols<-unlist(respCols)
  if(combResp=="mean")
    responsesComb<-rowMeans(responses[,respCols])
  else
    responsesComb<-rowMeans(round(responses[,respCols]))
}
print(proc.time()-start)
solutions<-data.frame(Problema=problemDirs,Soluzione=numeric(length(problemDirs)),stringsAsFactors=FALSE)#contiene risposte
solutions[,2]<-responsesComb
solutions[,2]<-formatC(round(solutions[,"Soluzione"],3),format="f",digits=3)
write.table(solutions,file=paste0(outputPath,"answers.txt"),quote =
              FALSE,row.names=FALSE,col.names=FALSE)