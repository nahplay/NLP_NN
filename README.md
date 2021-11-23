Here Is the implementation of LTSM and BLTSM neural networks and pretrained embeddings Word2vec and Glove.
1. The results are pretty similar to what linear models give
2. WordClouds for both positive and negative corpuses are almost identicall, yet there are some differiences which basically give us good results
3. It is quite usefull to have early stopping to achieve the best model(per val_loss,vall_accuracy etc.)and model checkpoints to save it.
4. Per learning curve visualization we can see at which epoch we should stop and not overfit the model.
5. LTSM models are good at pattern recognition.
