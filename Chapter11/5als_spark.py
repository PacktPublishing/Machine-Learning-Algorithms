import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS

# For reproducibility
np.random.seed(1000)

nb_users = 200
nb_products = 100
ratings = []

if __name__ == '__main__':
    conf = SparkConf().setAppName('ALS').setMaster('local[*]')
    sc = SparkContext(conf=conf)

    for _ in range(10):
        for i in range(nb_users):
            rating = Rating(user=i, product=np.random.randint(1, nb_products), rating=np.random.randint(0, 5))
            ratings.append(rating)

    # Parallelize the ratings
    ratings = sc.parallelize(ratings)

    # Train the model
    model = ALS.train(ratings, rank=5, iterations=10)

    # Test the model
    test = ratings.map(lambda rating: (rating.user, rating.product))

    predictions = model.predictAll(test)
    full_predictions = predictions.map(lambda pred: ((pred.user, pred.product), pred.rating))

    # Compute MSE
    split_ratings = ratings.map(lambda rating: ((rating.user, rating.product), rating.rating))
    joined_predictions = split_ratings.join(full_predictions)
    mse = joined_predictions.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean()

    print('MSE: %.3f' % mse)

    # Perform a single prediction
    prediction = model.predict(10, 20)
    print('Prediction: %.3f' % prediction)
