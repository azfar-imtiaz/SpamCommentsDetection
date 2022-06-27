# SpamCommentsDetection
Train a Machine Learning model using the YouTube Spam Collection Dataset, and perhaps some manually created dataset(s), to identify spam comments on YouTube videos

## Dataset
The dataset being used originally is the YouTube Spam Collection Dataset from UCI. It contains 5 CSV files, each containing approximately ~350 or ~450 rows. This data contains comments on each video, along with their classification (0 signifies ham and 1 signifies spam), YouTube comment ID, author of the comment, and the published date of the comment.

### Manually created dataset
Altogether, the 5 CSV files in the YouTube Spam Collection dataset is a very small dataset, and a classification model trained on this dataset will not scale well on comments from other YouTube videos (more on this later). Therefore, I have been experimenting with the YouTube API v3 from Google Console to scrape comments from other YouTube videos, in order to enlarge the dataset. The current plan is to scrape 350 - 450 comments from several YouTube videos.

The planned approach here is a semi-supervised one: train a classification model on the YouTube Spam Collection Dataset, and then use this model to classify comments in the YouTube scraped dataset. Initial attempts made in this regard have revealed a flaw: the classifier trained on the YouTube Spam Collection dataset does not perform very well on the scraped comments - specifically, the problem is with false positives. An initial observation is that comments that express some form of personal opinion are generally identified as spam.

Another problem that I have identified is that scraping comments from YouTube videos mostly yields ham comments (even when the comments are ordered by time instead of relevance). This is perhaps because spam comments are automatically filtered out by YouTube, or manually by the video creator. Maybe scraping comments down to the last pages might yield some more spam comments (need to investigate).

## Topic Modeling
One potential approach that I'm currently mentally grappling with is to use topic modeling to identify the major topics associated with a video (in an attempt to get what the video is about), and then somehow incorporate the topic distribution as a feature into the classification process of an unseen comment. There are several things to figure out here: 
- How many topics to use - should I use 2 topics? One for words relevant to the video topic, and the other for irrelevant ones? Or perhaps there can be more topics, and we have a TF-IDF weight for each word, so we get the sum of weights for say, top 5 words in each topic, and a higher score signifies a more "relevant" topic (although if we're using TF-IDF scores, a lower score would indicate higher relevance). Could also use a pre-trained word embedding model to get semantic similarity here.
- Topic Modeling cannot be used generally - we would have to use it either on a video level or a channel level. This is because the topics in one video (about tech, for example) won't have the same or even similar topics as a video about cats. So we can either do topic modeling on the comments for each individual video, or for all the videos in a single channel. The latter is something that can be experimented with because a channel about tech videos would still have similar topics... Needs to be experimented with.

Here is a link to follow: https://www.kaggle.com/code/nbuhagiar/spam-detection-with-topic-modelling/notebook

## Feature Extraction
I am using a fairly simple approach to feature extraction right now - TF-IDF vectors. Can incorporate topic distribution into the feature vectors later. Maybe word vectors from a pre-trained word embedding model could also be used, although incorporating slang, misspelled words, and proper nouns would probably be an issue. Finetune it then, if needed?

The bigger focus for feature extraction is text preprocessing, as the text data in YouTube comments is very messy. Emoji, misspelled words, slang, HTML elements in scraped comments. The cleaner this text is, the better. The current preprocessing function I have is a work in progress. I want to incorporate stemming or lemmatization there as well. Maybe also use something for correcting misspelled words, if something like that already exists?

## Classification
Currently, I have a pretty simple classification approach. I am training a Random Forest Classifier with 200 estimators. I will for sure experiment with different classification models. I would like to try out neural networks and/or deep learning as well, but I don't think I have enough data for that (yet). However, I think the main focus should not be on the classification algorithms, but rather the text preprocessing and feature extraction. 