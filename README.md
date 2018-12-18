# movie-recommendation-system

Recommend movies to users based on different collaborative filtering approaches.

This project used MovieLens - 100K dataset for predictions.
Link to dataset: http://grouplens.org/datasets/movielens/100k/
Information about the dataset can be found here: http://files.grouplens.org/datasets/movielens/ml-100k-README.txt

In order to run the code, the python file should be stored in the same folder where the dataset resides.
Command for running code:
>> python main.py - -data - -item

--data expects path to u.data file in the movie - lens dataset.
u.data: The full u data set, 100000 ratings by 943 users on 1682 items. his is a tab separated list of
user id | item id | rating | timestamp

--item expects path to u.item file in the movie - lens dataset.
u.item: Information about the items(movies). This is a tab separated list of
movie id | movie title | release date | video release date |
IMDb URL | unknown | Action | Adventure | Animation |
Children's | Comedy | Crime | Documentary | Drama | Fantasy |
Film - Noir | Horror | Musical | Mystery | Romance | Sci - Fi |
Thriller | War | Western |

Four different output files are generated in the folder 'output_files':
Item - based Cosine(cosine_predictions_item.csv)
User - based Cosine(cosine_predictions_user.csv)
SVD(svd_predictions.csv)

