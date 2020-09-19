# Author Verification

Approach used for 2015 PAN competition for the author identification task (https://pan.webis.de/clef15/pan15-web/authorship-verification.html)

Here you can find both the publication and the implementation of the approach proposed.

For the competition it is necessary to deliver a software that takes as input the documents and returns the answers about "Given two documents, are they written by the same author?".
For this reason it is used a training set of documents in order to create the trained models (implementation in the file [training.R](https://github.com/postaxmi/authorverification/blob/master/training.R)).
The trained models are saved and then they are used by the software that has been delivered (together with the trained models) to PAN for the competition; the implementation is in the file [applying.R](https://github.com/postaxmi/authorverification/blob/master/applying.R)

