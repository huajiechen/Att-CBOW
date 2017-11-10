## Attention Model for Continuous Bag of Words

### Requiement
- Python2.7
- Tensorflow0.12

### Data
Data format is the same with **gensim.Word2Vec**. Each line is an document. And the text should be segmente in advance.
Data in my experiment can be download from my [homepage](https://huajiechen.github.io/).

### Parameters
- filename: location of input fille
- batch_size: size of batch
- num_epochs: train times
- window_size: context window size
- min_time: minimum times of word
- embedding_size: embedding size of word and character
- num_sampled: negtive sampled size
- start_lr:learning rate