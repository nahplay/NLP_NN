from MovieDS import MovieDS
import tensorflow as tf

# GPU usage
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # The task itself
    dataset = MovieDS('LargeMovieReviewDataset.csv')
    prep_dataset = dataset.prep(dataset.eda())  # Storing preprocessed dataset after EDA
    dataset.visualization_popular(prep_dataset)  # Visualization of popular words from preprocessed dataset
    stemming_data = dataset.stemming_wc(prep_dataset)  #
    dataset.liner_model(*dataset.train_test_split(stemming_data))  # Dealing with linear models
    dataset.rnn_models(*dataset.train_test_split(stemming_data))  # Neural networks per task


if __name__ == "__main__":
    main()
