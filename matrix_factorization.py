#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# Model Tabanlı İşbirliği Filtreleme: Matris Faktörizasyonu, öneri sistemlerinde kullanılan bir tekniktir ve
# kullanıcıların geçmiş davranışları ve etkileşimde bulundukları öğeler temel alınarak kullanıcı tercihlerini veya
# puanlamalarını tahmin etmeyi amaçlar. Genel anlamıyla işbirliği filtreleme kavramına ait olan bu yöntem,
# kişiselleştirilmiş öneriler üretmek için yaygın bir yaklaşımdır.

# Model tabanlı işbirliği filtreleme, kullanıcıları ve öğeleri matematiksel bir model veya temsil şeklinde oluşturma
# fikrine dayanır ve genellikle bir matris şeklinde ifade edilir. Bu matris kullanım matrisi veya puan matrisi olarak
# bilinir, satırlar kullanıcıları, sütunlar ise öğeleri temsil eder ve her hücre, bir kullanıcının belirli bir öğe için
# verdiği puanı veya tercihi temsil eder. Ancak, bu matris genellikle seyrek bir yapıya sahiptir çünkü çoğu kullanıcı
# tüm öğeleri puanlamamış veya etkileşimde bulunmamıştır.

# Matris faktörizasyonu, kullanım matrisini genellikle kullanıcı matrisi ve öğe matrisi olarak adlandırılan daha düşük
# boyutlu iki matrise ayırma amacına yönelik bir yaklaşımdır. Her kullanıcı, gizli faktörlerin (özelliklerin) bir
# vektörü ile temsil edilir ve her öğe de başka bir vektör ile temsil edilir. Matris faktörizasyonunun fikri, gizli
# faktörlerin kullanıcıların ve öğelerin tercihlerini etkileyen temel özellikleri yakalamasını sağlamaktır.

# Kullanım matrisinin faktörizasyonuyla, model eksik puanları tahmin etmeyi veya yeni öğeler için puanlamalar yapmayı
# öğrenirken kullanılır. Bunun için, ilgili kullanıcı ve öğe vektörlerini çarparak gerçek puanlar ile tahmin edilen
# puanlar arasındaki hata miktarını minimize etmeyi hedefleyen bir optimizasyon süreci kullanılır.

# Model eğitildikten sonra, belirli bir kullanıcı için yüksek tahmin edilen puanlara sahip ancak kullanıcı tarafından
# daha önce etkileşimde bulunulmamış öğeleri tanımlayarak öneriler yapabilir. Bu, sistemin benzer kullanıcı tercihlerine
# dayanarak ilgili ve kişiselleştirilmiş öneriler sunmasını sağlar.

# Matris faktörizasyonunu kullanan model tabanlı işbirliği filtreleme, seyreklik sorununu ele almak ve çeşitli
# alanlarda, örneğin film, kitap, müzik ve e-ticaret gibi, doğru öneriler sağlamak için başarılı olmuştur. Gerçek dünya
# uygulamalarında yaygın olarak kullanılmakta olup, öneri sistemlerinde temel tekniklerden biri olarak kabul
# edilmektedir.


# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






