##########################################################
## Create edx set, validation set (final hold-out test set) ##
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




##########################################################
## Exploratory Data Analysis ##
##########################################################

# Getting Head of training dataset
head(edx) %>%
  print.data.frame()

# Getting the descriptive statistics for each column of the edx dataset
summary(edx)

# Getting the number of unique movies and users in the edx dataset 
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

#By using the code below, we can see that not every unique user rates every movies
keep<-edx %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
tab<-edx %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()


# Plot frequencies of each Ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "grey") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Number of each Ratings")


# Plot distribution of frequency of ratings per movie
# From the plot below it can be seen that the number of movie rated will decrease exponentially as the frequency of ratings increase
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 25, color = "grey")+
  xlab("Number of Ratings")+ylab("Number of Movies")+
  ggtitle("Number of Ratings in a Movie")

#To see a better relationship, we use log transformation on the number of ratings(x axis)
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 25, color = "grey")+
  scale_x_log10()+
  xlab("Log Transformed Number of Ratings")+ylab("Number of Movies")+
  ggtitle("Number of Ratings in a Movie")+theme_light()


# There are a considerable amount of movies that is only rated once
# Table below shows 20 movies that is rated only once
edx%>%group_by(movieId,title)%>%
  summarize(count=n(),rating=rating)%>%arrange(count)%>%
  head(20)%>%knitr::kable()



# Table that shows the top 20 number of ratings given by users
edx%>%group_by(userId)%>%
  summarize(count=n())%>%arrange(desc(count))%>%
  head(20)%>%knitr::kable()

edx%>%group_by(movieId,title)%>%
  summarize(count=n(),avg_rating=mean(rating))%>%arrange(desc(count))%>%
  head(20)%>%knitr::kable()

# Plot distribution of frequncy of ratings given by users
# From the plot below it can be seen again that the number of users will decrease exponentially as the number of rates given increase
edx%>%count(userId)%>%
  ggplot(aes(n))+geom_histogram(bins=25,color='grey')+
  xlab('Number of ratings given')+ylab('number of Users')+
  ggtitle("Number of Rating given by a User")

#To see a better relationship, we use log transformation on the number of ratings(x axis)
edx%>%count(userId)%>%
  ggplot(aes(n))+scale_x_log10()+geom_histogram(bins=25,color='grey')+
  xlab('Log Transformation of Number of ratings given')+ylab('number of users')+
  ggtitle("Number of Rating given by a User")+theme_light()




# Plot distribution of mean movie ratings given by users
# To avoid outliers and high bias, only users with more than 100 movies rated are used
# It can be seen from the plot below that most user give an average rating of between 3 and 4
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(average = mean(rating)) %>%
  ggplot(aes(average)) + geom_histogram(bins = 25, color = "grey") +
  xlab("Average rating") + ylab("Number of users") +
  ggtitle("Average movie ratings given by users") +
  theme_light()


##########################################################
## Modelling with Least Square Error and Regularization ##
##########################################################

# Create training and test set #

# make a new feature 'year' and 'month' which indicate year after 1995 and month of the movie
edx<-edx%>%
  mutate(year=year(as.POSIXct(timestamp,origin='1970-01-01'))-1994,
         month=month(as.POSIXct(timestamp,origin='1970-01-01')))
validation<-validation%>%
  mutate(year=year(as.POSIXct(timestamp,origin='1970-01-01'))-1994,
         month=month(as.POSIXct(timestamp,origin='1970-01-01')))


# Using 8:2 ratio for train and test set from edx dataset
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId, movieId, year, and movie in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "year") %>%
  semi_join(train_set, by = "month")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(removed,temp,test_index)

# Naive Average movie rating model #

# Compute the dataset's mean rating
mu <- mean(train_set$rating)
mu

# Test results based on naive prediction
# a fairly high RMSE (>1) which indicates a poor performance on the model 
# which make sense since no variable are taken into account in the model and every movie are rated the same
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse


# Creating and saving prediction in data frame, this dataframe will also be used in further models
rmse_results <- data_frame(method = "Naive Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()


# Movie effect model #

# Taking into account the movie effect b_i
# Obtained by getting the average of subtracting every rating received in a movie by the mean in previous result
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_avgs


# Plot number of movies with the computed b_i
movie_avgs %>% ggplot(aes(b_i))+geom_histogram(bins=25,color="grey")+
  xlab('movie effect coefficient b_i')+ylab('Number of Movies')+
  ggtitle("Number of Movies For Every Computed b_i")+theme_light()


# Prediction are done by adding the mean with the b_i corresponding to the movieID in test_set
# Test and save rmse results 
predicted_ratings_1 <- mu +  test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings_1, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse ))

# The RMSE obtained decreased from larger than 1 to 0.9437
rmse_results %>% knitr::kable()
  
  
# Movie and user effect model #
  
# Obtained by getting the average of subtracting every rating received in a movie by the mean and b_i in previous result
# Plot penalty term user effect using users that have rated more than 100 movies
# It can be seen below that the distribution of computed b_u is similar to normal distribution
train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))%>%
  ggplot(aes(b_u))+geom_histogram(bins=25,color="grey")+
  xlab('User effect coefficient b_u')+ylab('Number of Movies')+
  ggtitle("Number of Movies For Every Computed b_u")+theme_light()
  
  
#Saving the computed b_u for overall movies and users
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
  
  
# Prediction are done by adding the mean with the b_i corresponding to the movieID 
# and b_u corresponding to the userID in test_set
# Test and save rmse results 
predicted_ratings_2 <- test_set%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
  
model_2_rmse <- RMSE(predicted_ratings_2, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model",
                                     RMSE = model_2_rmse))
  
# The RMSE obtained become better with the value 0.865
rmse_results %>% knitr::kable()
  

# Movie, user, and release year effect model #

# Using similar approach as previous models
#Saving the computed b_y for overall movies, users, and years
year_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId')%>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))


# Prediction are done by adding the mean with the b_i corresponding to the movieID, 
# b_u corresponding to the user ID
# and b_y corresponding to the year in test_set 
# Test and save rmse results 
predicted_ratings_3 <- test_set%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs,by='year')%>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings_3, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie, user, year effect model",
                                     RMSE = model_3_rmse))

# The RMSE obtained has become slightly better with decrease of 0.000003
rmse_results %>% knitr::kable()


# Movie, user, year, month effect model #

# Using similar approach as previous models
#Saving the computed b_y for overall movies, users, years, and months
month_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId')%>%
  left_join(year_avgs, by='year')%>%
  group_by(month)%>%
  summarize(b_m = mean(rating - mu - b_i - b_u - b_y))


# Prediction are done by adding the mean with the b_i corresponding to the movieID, 
# b_u corresponding to the user ID, b_y corresponding to the year, and
# b_m corresponding to the month in test_set 
# Test and save rmse results 
predicted_ratings_4 <- test_set%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs,by='year')%>%
  left_join(month_avgs, by='month')%>%
  mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
  pull(pred)

model_4_rmse <- RMSE(predicted_ratings_4, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie, user, year, month effect model",
                                     RMSE = model_4_rmse))

# The RMSE obtained has become slightly better with decrease of 0.000003
rmse_results %>% knitr::kable()


# Regularized movie effect model #
  
# Ultilizing cross-validation to choose lambda, the tuning parameter
lambdas <- seq(0, 10, 0.25)
  
  
# For each lambda,find b_i, followed by rating prediction for validation & tested it using RMSE
rmses_bi <- sapply(lambdas, function(l){
  b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))
    
  predicted_ratings<-test_set%>%
    left_join(b_i, by='movieId') %>% 
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
  })
  
# using the qplot it can be seen that the lambda that returns the smallest RMSE is around 2.5
qplot(lambdas, rmses_bi,ylab="RMSE",xlab="Lambda",
      main="RMSEs of Regularized Movie Effect Model")
# The exact value of lambda can be seen below which is 2.5
lambdas[which.min(rmses_bi)]

#Saving results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie effect model",  
                                    RMSE = min(rmses_bi)))
# The result has a small improvement from the movie effect model
rmse_results %>% knitr::kable()
  
  
# Regularized Movie and User Effect Model #
  
# For each lambda, find b_i and b_u, followed by rating prediction for validation & tested it using RMSE
rmses_bu <- sapply(lambdas, function(l){
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
    
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>% group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
    
  return(RMSE(predicted_ratings, test_set$rating))
})
  
  
# using the qplot it can be seen that the lambda that returns the smallest RMSE is around 5
qplot(lambdas, rmses_bu,ylab="RMSE",xlab="Lambda",
      main="RMSEs of Regularized Movie and User Effect Model")  
  
# The exact optimal lambda can be seen below which is 4.75
lambdas[which.min(rmses_bu)]
  
# Saving results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                      RMSE = min(rmses_bu)))

# It can be seen in the table below that there is a slight improvement in the 
# regularized movie user effect model compared to the previous movie and user effect model
# with the decrease as big as 0.0007
rmse_results %>% knitr::kable()
  

# Regularized Movie, User, Year, Month Effect Model #

# For each lambda, find b_i, b_u, b_y, b_m 
# followed by rating prediction for validation & tested it using RMSE
rmses_fin <- sapply(lambdas, function(l){
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>% group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- train_set %>% 
    left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>%
    group_by(year) %>% summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l))
  
  b_m <- train_set %>% 
    left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>% group_by(month) %>%
    summarize(b_m = sum(rating - mu - b_i - b_u - b_y)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_m, by = "month") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})


# using the qplot it can be seen that the lambda that returns the smallest RMSE is around 5
qplot(lambdas, rmses_fin,ylab="RMSE",xlab="Lambda",
      main="RMSEs of Regularized Movie, User, year, and month Effect Model")  

# The exact optimal lambda can be seen below which is 5
lambdas[which.min(rmses_fin)]

# Saving results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie, User, year, and month Effect Model",  
                                     RMSE = min(rmses_fin)))

# It can be seen in the table below that there is a slight improvement in the 
# final regularized model with a decrease of 0.0007 compared to the unregularized one
rmse_results %>% knitr::kable()

  
#### Results ####                                                            
# Final model training with lambda=5 and edx dataset

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+5))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+5))

b_y <- edx %>% 
  left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>%
  group_by(year) %>% summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+5))

b_m <- edx %>% 
  left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>% group_by(month) %>%
  summarize(b_m = sum(rating - mu - b_i - b_u - b_y)/(n()+5))

prediction_ratings_fin <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "year") %>%
  left_join(b_m, by = "month") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_m) %>%
  pull(pred)

model_fin_rmse<-RMSE(prediction_ratings_fin,validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final model on validation dataset",  
                                     RMSE = model_fin_rmse))

# RMSE results overview 
rmse_results %>% knitr::kable()
  
#### Appendix ####
print("Operating System:")
version
  
  
  
  
  