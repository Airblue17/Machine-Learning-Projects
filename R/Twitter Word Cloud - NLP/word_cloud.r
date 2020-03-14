library(tm)
library(twitteR)
library(wordcloud)
library(webshot)
library(RColorBrewer)
library(qdapRegex)

# Connect to Twitter
setup_twitter_oauth(ckey, csecret, akey, asecret) # Access key and Consumer key already set


# Search Twitter 
bernie.tweets <- searchTwitter("bernie", n=2000, lang="en")
bernie.text <- sapply(bernie.tweets, function(x) x$getText())


# Remove NA rows and strip urls from the tweets
bernie.text <- bernie.text[!is.na(bernie.text)]


bernie.text <- rm_url(bernie.text)

bernie.text <- rm_twitter_url(bernie.text)

# Cleaning the data
bernie.text <- iconv(bernie.text, 'UTF-8', 'ASCII') # remove emoticons
bernie.corupus <- Corpus(VectorSource(bernie.text)) # create a corpus

# Creating a document term matrix
term.doc.matrix <- TermDocumentMatrix(bernie.corupus,
                                      control = list(removePunctuation = TRUE,
                                                     stopwords = c("bernie","sanders","RT","http", stopwords("english")),
                                                     wordLengths=c(5,Inf),
                                                     removeNumbers = TRUE,tolower = TRUE))




term.doc.matrix <- as.matrix(term.doc.matrix)

head(term.doc.matrix)

# Getting the word counts

word.freqs <- sort(rowSums(term.doc.matrix), decreasing=TRUE) 
dm <- data.frame(word=names(word.freqs), freq=word.freqs)


# Creating the word cloud

path = 'R/Machine Learning Projects/R/Twitter Word Cloud - NLP/plots/'

jpeg(paste(path, "wordcloud.jpg", sep = ""))

wordcloud(dm$word, dm$freq, random.order=FALSE, colors=brewer.pal(8, "Dark2"))

title("World Cloud of tweets with the word 'Bernie' in it")
dev.off()

