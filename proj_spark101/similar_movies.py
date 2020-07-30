from pyspark import SparkContext
from math import sqrt
import sys


def remove_duplicates(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


def make_movie_pairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))


def compute_score(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

if __name__ == "__main__":
    sc = SparkContext()

    data = sc.textFile("hdfs://devenv/user/spark/spark101/movies_similarity/data/movie_ratings")

    user_movie_ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

    self_joined_ratings = user_movie_ratings.join(user_movie_ratings)

    distinct_self_joined_ratings = self_joined_ratings.filter(remove_duplicates)

    movie_pairs = distinct_self_joined_ratings.map(make_movie_pairs)

    movie_pair_ratings = movie_pairs.groupByKey()

    movie_pair_with_scores = movie_pair_ratings.mapValues(compute_score)

    movies_similarity_result = movie_pair_with_scores.collect()

    for movie_pair in movies_similarity_result:
        print(movie_pair)
