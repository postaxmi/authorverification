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
  result<-gsub("[[:alpha:]]*[[:upper:]][[:alpha:]]*[[:upper:]][[:alpha:]]*","w",result)#parola con 2 o pi?? maiuscole
  result<-gsub("[[:upper:]][[:lower:]]*","u",result)#parola con una maiuscola all'inizio (o solo lettera maiuscola)
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
  result<-data.frame(Symbol=all,Mean=numeric(length(all)),Dev=numeric(length(all)),Occurrences=numeric(length(all)))
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
  return ((treetag(file,treetagger="~/treetagger/cmd/tree-tagger-spanish", lang="es"))@TT.res$tag)
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
  #  print(x)
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
  correct<-nrow(instances[instances$GrandeVeritas==1 & instances[,1]<threshold,])+nrow(instances[instances$GrandeVeritas==0 & instances[,1]>=threshold,])
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

#usare questa prima parte piuttosto che la vecchia essendo cambiata struttura dataset
inputPath<-"/home/postaxmi/pan2/panSP/"
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
#language="EN"
#root<-"/home/q/pan/"# mettere percorso assoluto per la lingua greca
#path<-paste(root,language,"/",sep="") #path of the corpus
#problemDirs<-list.files(path) #directories of all problems
allDocs<-list() #global list with all documents in the corpus
allDocsPOSed<-list()#global list with all documents transformed to POS in the corpus


#language="DU"
#root<-"/home/q/pan/"# mettere percorso assoluto per la lingua greca
#path<-paste(root,language,"/",sep="") #path of the corpus
#problemDirs<-list.files(path) #directories of all problems
#allDocs<-list() #global list with all documents in the corpus
#allDocsPOSed<-list()#global list with all documents transformed to POS in the corpus

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
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOS(allDocs[[paste(problem,document,sep='-')]],sent_token_annotator,word_token_annotator,pos_tag_annotator),collapse=" ")
    }
  }
}else if(language=="SP"){
  for(problem in problemDirs){ #for all problems
    print(problem)
    documents<-list.files(paste(path,problem,sep=""))#documents for this problem
    for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOSES(paste(path,problem,"/",document,sep="")),collapse=" ")
    }
  }
}else if(language=="GR"){
  for(problem in problemDirs){ #for all problems
    print(problem)
    documents<-list.files(paste(path,problem,sep=""))#documents for this problem
    for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
      allDocs[[paste(problem,document,sep='-')]]<-paste(readLines(paste(path,problem,"/",document,sep="")),collapse=" ")#read document
      allDocsPOSed[[paste(problem,document,sep='-')]]<-paste(getPOSGR(paste(path,problem,"/",document,sep="")),collapse=" ")
    }
  }
}
print(proc.time()-start)#470s 100 EN problems

start<-proc.time()
#get ngram (word, char, POS) of all documents
#occurrences of word ngram for all documents
minWordNgrams<-1
maxWordNgrams<-3
maxWordNgramsLength<-500#massimo numero di word ngrams considerati (per specifico n)
wordNgramsTot<-list()#contiene tutti ngrams word
for(i in minWordNgrams:maxWordNgrams){
  wordNgramsTot[[i]]<-getWordNgrams(allDocs,i)#textcnt(allDocs,n=i,method="string")
}
wordNgrams<-unlist(wordNgramsTot)#contiene ngrams word che superano selezione
if(maxWordNgramsLength>0){
  wordNgrams<-sort(wordNgrams,decreasing=TRUE)[1:maxWordNgramsLength]
}else{
  wordNgrams<-sort(wordNgrams,decreasing=TRUE)
}
#occurrences of char ngram for all documents
maxCharNgrams<-5
maxCharNgramsLength<-500#massimo numero di char ngrams considerati (per tutti n)
charNgramsTot<-getCharNgrams(allDocs,maxCharNgrams)#textcnt(allDocs,n=maxCharNgrams,method="ngram",decreasing=TRUE,tolower=FALSE)
if(maxCharNgramsLength>0){
  charNgrams<-charNgramsTot[1:maxCharNgramsLength]
}else{
  charNgrams<-charNgramsTot
}
minPOSNgrams<-1
maxPOSNgrams<-3
maxPOSNgramsLength<-500#massimo numero di word ngrams considerati (per specifico n)
POSNgramsTot<-list()
for(i in minPOSNgrams:maxPOSNgrams){
  POSNgramsTot[[i]]<-getPOSNgrams(allDocsPOSed,i)# textcnt(allDocsPOSed,split=" ",n=i,method="string")
}
POSNgrams<-unlist(POSNgramsTot)
if(maxPOSNgramsLength>0){
  POSNgrams<-sort(POSNgrams,decreasing=TRUE)[1:min(length(POSNgrams),maxPOSNgramsLength)]
}else{
  POSNgrams<-sort(POSNgrams,decreasing=TRUE)
}
maxPunctuationNgrams<-2
punctuationNgramsTot<-getPunctNgrams(allDocs,maxPunctuationNgrams)
maxTextShapeNgrams<-3
maxTextShapeLength<-100
textShapeNgramsTot<-getNgramsShape(allDocs,maxTextShapeNgrams)
if(maxTextShapeLength>0){
  textShapeNgramsTot<-sort(textShapeNgramsTot,decreasing=TRUE)[1:min(length(textShapeNgramsTot),maxTextShapeLength)]
}
write(names(wordNgrams),file=paste0("~/W",language,maxWordNgramsLength))
write(names(charNgrams),file=paste0("~/C",language,maxCharNgramsLength))
write(names(POSNgrams),file=paste0("~/P",language,maxPOSNgramsLength))
write(names(punctuationNgramsTot),file=paste0("~/p",language))
write(names(textShapeNgramsTot),file=paste0("~/ts",language,maxTextShapeLength))
print(proc.time()-start)

#for each problem get occurences for all ngram retrieved previously
start<-proc.time()
problems<-list()
#lunghezza di ogni documento
lengthDoc<-data.frame(Problema=character(0),NomeDoc=character(0),TotalLength=numeric(0))
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
  problems[[problem]]<-lapply(1:8,function(x,y) data.frame(NomeDoc=y,stringsAsFactors=FALSE),y=documents)
  problems[[problem]][[1]][,names(wordNgrams)]<-0
  problems[[problem]][[2]][,names(charNgrams)]<-0#add as columns all char ngrams retrieved
  #add as columns all POS ngrams retrieved
  problems[[problem]][[3]][,names(POSNgrams)]<-0
  problems[[problem]][[4]][,paste0("Wl#",as.character(1:50))]<-0#per lunghezza parole
  problems[[problem]][[5]][,paste0("Sl#",as.character(1:120))]<-0#per lunghezza sentences
  problems[[problem]][[6]][,"Wr"]<-0#per word richness
  problems[[problem]][[7]][,names(punctuationNgramsTot)]<-0#per punctuation ngrams
  problems[[problem]][[8]][,names(textShapeNgramsTot)]<-0#per textshape ngrams
  for(document in documents){#for all documents of this problem
    #word ngrams
    wordNgramsDoc<-unlist(lapply(minWordNgrams:maxWordNgrams,function(x,y) getWordNgrams(y,x),y=allDocs[[paste(problem,document,sep="-")]]))
    columnsTmp<-intersect(names(wordNgramsDoc),names(wordNgrams))
    problems[[problem]][[1]][problems[[problem]][[1]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=wordNgramsDoc)
    #lengthDoc<-rbind(lengthDoc,data.frame(Problema=problem,NomeDoc=document,TotalLength=sum(problems[[problem]][[1]][problems[[problem]][[1]]$NomeDoc==document,2:ncol(problems[[problem]][[1]])]),stringsAsFactors=FALSE))
    #char ngrams
    charNgramsDoc<-getCharNgrams(allDocs[[paste(problem,document,sep="-")]],maxCharNgrams)#textcnt(allDocs[[paste(problem,document,sep="-")]],method="ngram",n=maxCharNgrams,tolower=FALSE)#char ngrams for this document
    columnsTmp<-intersect(names(charNgramsDoc),names(charNgrams))         	 
    problems[[problem]][[2]][problems[[problem]][[2]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=charNgramsDoc)
    #POS ngrams
    POSNgramsDoc<-unlist(lapply(minPOSNgrams:maxPOSNgrams,function(x,y) getPOSNgrams(y,x),y=allDocsPOSed[[paste(problem,document,sep="-")]]))
    columnsTmp<-intersect(names(POSNgramsDoc),names(POSNgrams))
    problems[[problem]][[3]][problems[[problem]][[3]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=POSNgramsDoc)
    #lengths of words and sentences
    #lengths<-getWordLengthCount(names(textcnt(allDocs[[paste(problem,document,sep="-")]],method="string",n=1L)))#lunghezza parole contando sulle parole uniche
    lengths<-getAllWordLengthCount(allDocs[[paste(problem,document,sep="-")]])#lunghezza parole contando su tutte le parole
    problems[[problem]][[4]][problems[[problem]][[4]]$NomeDoc==document,names(lengths)]<-lengths
    lengths<-getSentenceLengthCount(allDocs[[paste(problem,document,sep="-")]])
    problems[[problem]][[5]][problems[[problem]][[5]]$NomeDoc==document,names(lengths)]<-lengths
    #richness of vocabulary
    wR<-getWordsRichness(textcnt(allDocs[[paste(problem,document,sep="-")]],method="string",n=1L))
    problems[[problem]][[6]][problems[[problem]][[6]]$NomeDoc==document,names(wR)]<-wR
    #punctuation ngrams
    punctuationNgramsDoc<-getPunctNgrams(allDocs[[paste(problem,document,sep="-")]],maxPunctuationNgrams)
    columnsTmp<-intersect(names(punctuationNgramsDoc),names(punctuationNgramsTot))
    problems[[problem]][[7]][problems[[problem]][[7]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=punctuationNgramsDoc)
    #textshapeNgrams
    textShapeNgramsDoc<-getNgramsShape(allDocs[[paste(problem,document,sep="-")]],maxTextShapeNgrams)
    columnsTmp<-intersect(names(textShapeNgramsDoc),names(textShapeNgramsTot))
    problems[[problem]][[8]][problems[[problem]][[8]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=textShapeNgramsDoc)
  }
  for(l in colnames(sentencesLength)){
    sentencesLength[[l]]<-sentencesLength[[l]]+sum(problems[[problem]][[5]][,l])
  }
  totDocs<-totDocs+nrow(problems[[problem]][[5]])
  #relative frequencies
  for(i in 1:8){#for all type of ngrams
    if(i!=6){#no frequency for word richness, there's only a value
      nColumns<-ncol(problems[[problem]][[i]])
      for(r in 1:nrow(problems[[problem]][[i]])){#for all documents (rows)
        problems[[problem]][[i]][r,2:nColumns]<-prop.table(problems[[problem]][[i]][r,2:nColumns])
      }
    }
  }
}
print(proc.time()-start) #(234s 100 EN problems all unigrams)
#now problems[[p]] contains 3 dataframes for problem p: (1)word n-grams, (2) char n-grams (3) POS n-grams (each row is a document, name in column doc)
#transform each problem to an istance for a classifier

#sentencesLength contiene numero totale di frasi con lunghezza n, n=1,2,3...
#dividiamo in ng gruppi con pi?? o meno stessa cardinalit?? per avere gruppi di frasi in base a loro lunghezza
#ng<-4
#howManyEachGroup<-sum(sentencesLength[1,])/ng
#sentenceGroups<-data.frame(Group=numeric(ng),Value=numeric(ng))
#length<-0
#for(i in 1:ng){
#  sentenceGroups[i,"Group"]<-i
#  acc<-0
#  while(acc<howManyEachGroup && length<ncol(sentencesLength)){
#	length<-length+1
#	acc<-acc+sentencesLength[[1,paste0("Sl#",length)]]
#   # acc<-acc+sentencesLength[[1,paste0(length)]]
#  }
#  sentenceGroups[i,"Value"]<-length
#}

ng<-5
howManyEachGroup<-numeric(ng)
howManyEachGroup[[1]]<-sum(sentencesLength[1,])*0.1
howManyEachGroup[[2]]<-sum(sentencesLength[1,])*0.25
howManyEachGroup[[3]]<-sum(sentencesLength[1,])*0.75
howManyEachGroup[[4]]<-sum(sentencesLength[1,])*0.9
howManyEachGroup[[5]]<-sum(sentencesLength[1,])*1
sentenceGroups<-data.frame(Group=numeric(ng),Value=numeric(ng))
length<-0
acc<-0
for(i in 1:ng){
  sentenceGroups[i,"Group"]<-i
  while(acc<howManyEachGroup[[i]] && length<ncol(sentencesLength)){
    length<-length+1
    acc<-acc+sentencesLength[[1,paste0("Sl#",length)]]
    # acc<-acc+sentencesLength[[1,as.character(length)]]
  }
  sentenceGroups[i,"Value"]<-length
}
write.table(sentenceGroups,file=paste("Sg",language,sep=""))
#ora per ciascun documento etichettare le frasi in base al loro gruppo
allDocsSl<-list()
for(problem in problemDirs){ #for all problems
  print(problem)
  documents<-list.files(paste(path,problem,sep=""))#documents for this problem
  for(document in list.files(paste(path,problem,sep=""))){#for all documents of this problem
    allDocsSl[[paste(problem,document,sep='-')]]<-paste(getSentenceLengthGroups(allDocs[[paste(problem,document,sep='-')]],sentenceGroups),collapse="")
  }
}

ngramsSl<-list()
for(i in 1:ng){
  ngramsSl[[length(ngramsSl)+1]]<-paste0("S#",i)
  for(j in 1:ng){
    ngramsSl[[length(ngramsSl)+1]]<-paste0("S#",i,j)
  }
}
ngramsSl<-unlist(ngramsSl)

for(problem in problemDirs){ #for all problems
  print(problem)
  documents<-list.files(paste(path,problem,sep=""))#documents for this problem
  problems[[problem]][[9]]<-data.frame(NomeDoc=documents,stringsAsFactors=FALSE)
  #it will add a dataframes for sentence groups ngrams
  problems[[problem]][[9]][,ngramsSl]<-0#per sentences groups
  for(document in documents){#for all documents of this problem
    ngramsSlDoc<-getSentGroupNgrams(allDocsSl[[paste(problem,document,sep='-')]],2)#textcnt(allDocsSl[[paste(problem,document,sep='-')]],split="ngram",n=2L)
    columnsTmp<-intersect(names(ngramsSlDoc),ngramsSl)
    problems[[problem]][[9]][problems[[problem]][[9]]$NomeDoc==document,columnsTmp]<-lapply(columnsTmp,function(x,y) y[[x]],y=ngramsSlDoc)
  }
  #relative frequencies
  nColumns<-ncol(problems[[problem]][[9]])
  for(r in 1:nrow(problems[[problem]][[9]])){#for all documents (rows)
    problems[[problem]][[9]][r,2:nColumns]<-prop.table(problems[[problem]][[9]][r,2:nColumns])
  }
}

allInstFeats<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)
distinctInstFeats<-list()
#ground truth
truth<-read.table(paste(path,"truth.txt",sep=""),header=FALSE,stringsAsFactors=FALSE)
for(r in 1:nrow(truth)){#transform Y to 1 and N to 0
  truth[r,2]<-ifelse(truth[r,2]=="Y", 1, 0)
}
for(i in 1:9){
  allInstFeats[,colnames(problems[[1]][[i]])[-1]]<-0
  distinctInstFeats[[i]]<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)
  distinctInstFeats[[i]][,colnames(problems[[1]][[i]])[-1]]<-0
}
#add column with the truth
for(problem in allInstFeats[,1]){
  allInstFeats[allInstFeats$Problema==problem,"GrandeVeritas"]<-as.numeric(truth[truth[[1]]==problem,2])
  for(i in 1:length(distinctInstFeats)){
    distinctInstFeats[[i]][distinctInstFeats[[i]]$Problema==problem,"GrandeVeritas"]<-as.numeric(truth[truth[[1]]==problem,2])
  }
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
      kno[[i]]<-mean(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc!="unknown.txt",2:ncol(problems[[problem]][[i]])])
      distinctInstFeats[[i]][distinctInstFeats[[i]]$Problema==problem,"Wr"]<-abs(kno[[i]]-unk[[i]])
      allInstFeats[allInstFeats$Problema==problem,"Wr"]<-abs(kno[[i]]-unk[[i]])
    }else{
      unk[[i]]<-unlist(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc=="unknown.txt",2:ncol(problems[[problem]][[i]])])
      #aggregate all known document with weighted mean (the weight is the length of the document)
      kno[[i]]<-unlist(sapply(problems[[problem]][[i]][problems[[problem]][[i]]$NomeDoc!="unknown.txt",2:ncol(problems[[problem]][[i]])],mean))
      distinctInstFeats[[i]][distinctInstFeats[[i]]$Problema==problem,names(kno[[i]])]<-abs(kno[[i]]-unk[[i]])
      allInstFeats[allInstFeats$Problema==problem,names(kno[[i]])]<-abs(kno[[i]]-unk[[i]])
    }
  }
}
print(proc.time()-start)#(140s for 100 EN problems all unigrams)
proc.time()
repetitions<-100#repeat training and test
# training 50% test 50%
# trainingIndx<-createDataPartition(instances$GrandeVeritas,p=.5,list=FALSE,times=repetitions)
trainingIndx<-matrix(nrow=99,ncol=100)#leave one out
for(i in 1:100){
  trainingIndx[,i]<-seq(1:100)[-i]#c(i,(i+1)%%100)#
}
modelNames<-c("Glm","LogReg","Svm","SvmL","SvmP","TreeC","TreeR","RandF")
#modelNames<-c("","","Svm")
modelFeatNames<-list()
for(m in modelNames){
  for(f in 1:(1+length(distinctInstFeats))){
    modelFeatNames[length(modelFeatNames)+1]<-paste0(m,f)
  }
}
modelFeatNames<-unlist(modelFeatNames)
#conterr?? accuracy su trainining e test per ciascun modello, per ciascun tipo  di feature su cui si basa
accModels<-data.frame(Model=modelFeatNames,Training=numeric(length(modelFeatNames)),Test=numeric(length(modelFeatNames)))
#conterr?? accuracy su training e test per ciascun modello, raggruppando risultati ottenuti su diversi tipi di features, con media o voto
accEnsemble<-data.frame(Ensemble=modelNames,TrainingM=numeric(length(modelNames)),TestM=numeric(length(modelNames)),TrainingV=numeric(length(modelNames)),TestV=numeric(length(modelNames)))
for(t in 1:repetitions){
  print(paste("repetition",t))
  # trainingSet<-instances[trainingIndx[,t],2:ncol(instances)]
  # testSet<-instances[-trainingIndx[,t],2:ncol(instances)]
  #trainingSetMod<-allInstFeats[trainingIndx[,t],1:ncol(allInstFeats)]
  #testSetMod<-allInstFeats[-trainingIndx[,t],1:ncol(allInstFeats)]
  responses<-data.frame(Problema=problemDirs,stringsAsFactors=FALSE)#contiene risposte date dai vari modelli (uno per colonna) per ogni problema(uno per riga)
  trainingSetMod<-list()
  testSetMod<-list()
  for(f in 1:length(distinctInstFeats)){
    # trainingSetMod[[f]]<-distinctInstFeats[[f]][trainingIndx[,t],1:ncol(distinctInstFeats[[f]])]
    #testSetMod[[f]]<-distinctInstFeats[[f]][-trainingIndx[,t],1:ncol(distinctInstFeats[[f]])]
    trainingSetMod[[f]]<-distinctInstFeats[[f]][trainingIndx[,t],]
    testSetMod[[f]]<-distinctInstFeats[[f]][-trainingIndx[,t],]
  }
  trainingSetMod[[length(distinctInstFeats)+1]]<-allInstFeats[trainingIndx[,t],]
  testSetMod[[length(distinctInstFeats)+1]]<-allInstFeats[-trainingIndx[,t],]
  for(f in 1:length(trainingSetMod)){
    #  trainingSetMod<-distinctInstFeats[[f]][trainingIndx[,t],1:ncol(distinctInstFeats[[f]])]
    #  testSetMod<-distinctInstFeats[[f]][-trainingIndx[,t],1:ncol(distinctInstFeats[[f]])]
    colnames(trainingSetMod[[f]])[2:(ncol(trainingSetMod[[f]])-1)]<-paste0("F",seq(1:(ncol(trainingSetMod[[f]])-2)))
    colnames(testSetMod[[f]])[2:(ncol(testSetMod[[f]])-1)]<-paste0("F",seq(1:(ncol(testSetMod[[f]])-2)))
    #provvisorio per togliere colonne piene di zeri (influenza svm)
    if(f==4){
      trainingSetMod[[f]]<-trainingSetMod[[f]][,c(1:16,ncol(trainingSetMod[[f]]))]
    }
    if(f==5){
      trainingSetMod[[f]]<-trainingSetMod[[f]][,c(1:40,ncol(trainingSetMod[[f]]))]
    }
    #classificatori standards : logistic regression, svm(default, linear kernel and polynomial kernel), rpart (classification e regression)
    models<-list()
    models[[1]]<-glm(GrandeVeritas~.,data=trainingSetMod[[f]][,-1])#generalized linear model (gamily=gaussian)
    models[[2]]<-glm(GrandeVeritas~.,data=trainingSetMod[[f]][,-1],family=binomial)#logistic regression
    models[[3]]<-svm(GrandeVeritas~.,data=trainingSetMod[[f]][,-1])
    models[[4]]<-svm(GrandeVeritas~.,data=trainingSetMod[[f]][,-1],kernel="linear")
    models[[5]]<-svm(GrandeVeritas~.,data=trainingSetMod[[f]][,-1],kernel="polynomial")
    models[[6]]<-rpart(GrandeVeritas~.,data=trainingSetMod[[f]][,-1],method="class")
    models[[7]]<-rpart(GrandeVeritas~.,data=trainingSetMod[[f]][,-1],method="anova")
    models[[8]]<-randomForest(GrandeVeritas~.,data=trainingSetMod[[f]][,-1])
    for(i in 6:7){#prune trees
      models[[i]]<-prune(models[[i]], cp=0.1)#models[[i]]$cptable[which.min(models[[i]]$cptable[,"xerror"]),"CP"])
    }
    
    predictions<-list()
    for(i in 1:length(modelNames)){
      predictions[[i]]<-predict(models[[i]],trainingSetMod[[f]])
      if(i==6){#classification tree
        predictions[[i]]<-predictions[[i]][,"1"]
      }
      responses[trainingIndx[,t],paste0(modelNames[[i]],f)]<-predictions[[i]]
      #positive<-predictions[[i]]>0.5
      #accModels[[dist]][i,2]<-accModels[[dist]][i,2]+(length(which(positive&truth[trainingIndx[,t],2]==1))+length(which(!positive&truth[trainingIndx[,t],2]==0)))/nrow(trainingSetMod)
      accModels[accModels$Model==paste0(modelNames[[i]],f),2]<-accModels[accModels$Model==paste0(modelNames[[i]],f),2]+evaluateAccuracy(predictions[[i]],truth[trainingIndx[,t],2])
      predictions[[i]]<-predict(models[[i]],testSetMod[[f]])
      if(i==6){#classification tree
        predictions[[i]]<-predictions[[i]][,"1"]
      }
      responses[-trainingIndx[,t],paste0(modelNames[[i]],f)]<-predictions[[i]]
      # positive<-predictions[[i]]>0.5
      #accModels[i,3]<-accModels[[dist]][i,3]+(length(which(positive&truth[-trainingIndx[,t],2]==1))+length(which(!positive&truth[-trainingIndx[,t],2]==0)))/nrow(testSetMod)
      accModels[accModels$Model==paste0(modelNames[[i]],f),3]<-accModels[accModels$Model==paste0(modelNames[[i]],f),3]+evaluateAccuracy(predictions[[i]],truth[-trainingIndx[,t],2])
    }
  }
  #ora responses contiene una riga per ciascun problema e su ogni colonna la risposta data da un certo modello basato su una certo tipo di feature
  for(m in modelNames){
    respCols<-list()
    for(f in 1:length(distinctInstFeats)){
      respCols[length(respCols)+1]<-paste0(m,f)    
    }
    respCols<-unlist(respCols)
    resp<-rowMeans(responses[trainingIndx[,t],respCols])
    accEnsemble[accEnsemble$Ensemble==m,"TrainingM"]<-accEnsemble[accEnsemble$Ensemble==m,"TrainingM"]+evaluateAccuracy(resp,truth[trainingIndx[,t],2])
    resp<-rowMeans(responses[-trainingIndx[,t],respCols])
    accEnsemble[accEnsemble$Ensemble==m,"TestM"]<-accEnsemble[accEnsemble$Ensemble==m,"TestM"]+evaluateAccuracy(resp,truth[-trainingIndx[,t],2])
    resp<-rowMeans(round(responses[trainingIndx[,t],respCols]))
    accEnsemble[accEnsemble$Ensemble==m,"TrainingV"]<-accEnsemble[accEnsemble$Ensemble==m,"TrainingV"]+evaluateAccuracy(resp,truth[trainingIndx[,t],2])
    resp<-rowMeans(round(responses[-trainingIndx[,t],respCols]))
    accEnsemble[accEnsemble$Ensemble==m,"TestV"]<-accEnsemble[accEnsemble$Ensemble==m,"TestV"]+evaluateAccuracy(resp,truth[-trainingIndx[,t],2])
  }
  tmpAcc<-accModels
  tmpAcc[,2:3]<-tmpAcc[,2:3]/t
  print(tmpAcc)
  tmpAcc<-accEnsemble
  tmpAcc[,2:5]<-tmpAcc[,2:5]/t
  print(tmpAcc)
}
print(proc.time()-start)
accModels[,2:3]<-accModels[,2:3]/repetitions
accEnsemble[,2:5]<-accEnsemble[,2:5]/repetitions


#save models for random forest
models<-list()
for(f in 1:length(distinctInstFeats)){
  trainingSetMod[[f]]<-distinctInstFeats[[f]]
  colnames(trainingSetMod[[f]])[2:(ncol(trainingSetMod[[f]])-1)]<-paste0("F",seq(1:(ncol(trainingSetMod[[f]])-2)))
  #togliere colonne piene di zeri (influenza svm)
  if(f==4){
    trainingSetMod[[f]]<-trainingSetMod[[f]][,c(1:16,ncol(trainingSetMod[[f]]))]
  }
  if(f==5){
    trainingSetMod[[f]]<-trainingSetMod[[f]][,c(1:40,ncol(trainingSetMod[[f]]))]
  }
  models[[f]]<-randomForest(GrandeVeritas~.,data=trainingSetMod[[f]][,-1])
  saveRDS(models[[f]],file=paste0("rf",f,language))
}


