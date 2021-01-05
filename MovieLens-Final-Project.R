---
  title: "HarvardX PH125.9x Data Science Capstone-Movielens Project"
author: "Alvin Subakti"
date: "January 4, 2021"
output: pdf_document
---
  
  ```{r, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# 1. Introduction

## Background
A recommendation system is a method of information filtering system that has the capability to predict the preference of a user toward a certain item. Recommendation systems have been applied in a variety of areas including music, news, books, research articles, search queries, even in more advanced aspects like financial services, life insurances, and many more. One of them which is being applied in this project is regaring movie ratings. Movie rating recommendation system will be able to predict the rating a user would give toward a movie giving their traits.

Netflix realized the importance of this recommendation system in its movie rating system and held a open competition in 2006. In the competition, Netflix offered a million dollar prize to anyone that is able to imrpove the effectiveness of their recommendation system by 10%. This project is inspired by the competition. Here we are trying to solve a similiar problem but with a much simpler approach and a different dataset. Since the approach done by the participants in the competition are far more advanced and we may not have the capabilities or hardware requirements for it, so a simpler yet effective approach is used as a way to show understanding of this course. Also since the dataset used in the competition is not publicly available, This project will use another publicly available dataset related to movie ratings provided in the 'MovieLens'.

## DataSet
The Dataset used in this problem is the 'MovieLens' dataset, this dataset can be found and downloaded through these following links:
  
  https://grouplens.org/datasets/movielens/10m/
  
  http://files.grouplens.org/datasets/movielens/ml-10m.zip

Generation of 'Movielens' Dataset that will be used in this project will be loaded using the instructions given in the course.

## Goal
The goal of this project is to be able to analyze and gain insights from the 'Movielens' dataset. Then also able to construct machine learning models or algorithms that will be trained by using the training dataset, and finally has the ability to predict ratings given a movie in the validation dataset.

The parameter that will be used in this project is the RMSE or Rooted Mean Square Error. A model has a better performance if it has a smaller value of RMSES given the same validation dataset.


```{r, include=FALSE, echo=FALSE}
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
```


# 2. Exploratory Data Analysis
First, the prevew of the dataset can be seen in the following table, where the dataset consisted of user ID, Movie ID, movie ratings, timestamp, title of the movies, and genre of the movies.
```{r, include=TRUE, echo=FALSE}
head(edx) %>%
  print.data.frame()
```

The descriptive statistics for the dataset can be seen in the following table.
```{r, include=TRUE, echo=FALSE}
summary(edx)
```

Now, by using the following code we can see that the edx dataset generated contains 69878 unique users and 10677 unique movies.
```{r, echo=T,results='hide'}
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))
```

However, the amount of observation is not exactly $69878 \times 10677$ which indicated that not every users rate every movies in the dataset. This can be proven in the following table that show a user rate some of the movies shown by an existing rating but not all of the movies shown by the missing value NA.
```{r, include=TRUE, echo=FALSE}
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
``` 

##Distributions
Now the exploratory data analysis will continue by analyzing distributions in several aspects.

The distribution for the number of ratings given by all of the users in the edx dataset can be seen in the following plot. It is clear that most users give a rate of between 3 to 4 for a movie shown by the significant peak in this interval.
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%',warning=FALSE, message=FALSE}
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "grey") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Number of each Ratings")
```


Now for the validation dataset the rating distribution also can be seen in the plot below. By comparing the result of this two plot, we can see that edx and validation dataset has similar rating distribution.
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%',warning=FALSE, message=FALSE}
validation %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "grey") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Number of each Ratings (Validation)")
```


\pagebreak
Next, by plotting the distribution of the number of ratings given for a movie can be seen in the following plot
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 25, color = "grey")+
  xlab("Number of Ratings")+ylab("Number of Movies")+
  ggtitle("Number of Ratings in a Movie")
```

From the plot above it can be seen that every movies has different amount of rating given, and the number of movie rated has a tendency to decrease exponentially as the frequency of ratings increase. To observe a much better relationship, we will now plot the distribution of the number of ratings given for a movie but by also applying log transformation on the number of ratings (the x axis)
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 25, color = "grey")+
  scale_x_log10()+
  xlab("Log Transformed Number of Ratings")+ylab("Number of Movies")+
  ggtitle("Number of Ratings in a Movie")+theme_light()
```

The following tables shows the 20 least and most rated movies as well as the number of ratings given. We can see that there is a significant difference in the rating since there are movies that is only rated once and there are also blockbusters that has thousand of ratings.
```{r, include=TRUE, echo=FALSE,message=FALSE,warning=FALSE}
edx%>%group_by(movieId,title)%>%
  summarize(count=n(),rating=rating)%>%arrange(count)%>%
  head(20)%>%knitr::kable()
```

```{r, include=TRUE, echo=FALSE,warning=FALSE, message=FALSE}
edx%>%group_by(movieId,title)%>%
  summarize(count=n(),avg_rating=mean(rating))%>%arrange(desc(count))%>%
  head(20)%>%knitr::kable()
```

\pagebreak

Next, we will observe the distribution of the number of ratings given by a user. The following plot shows the relationship
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
edx%>%count(userId)%>%
  ggplot(aes(n))+geom_histogram(bins=25,color='grey')+
  xlab('Number of ratings given')+ylab('number of Users')+
  ggtitle("Number of Rating given by a User")
```

The previous founding regarding that not every user rate the same amount of movies is supported by the plot above. it can also be seen that the number of users has the tendency to decrease exponentially as the number of rates given increase. To see a better relationship, we use log transformation on the number of ratings(x axis).
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
edx%>%count(userId)%>%
  ggplot(aes(n))+scale_x_log10()+geom_histogram(bins=25,color='grey')+
  xlab('Log Transformation of Number of ratings given')+ylab('number of users')+
  ggtitle("Number of Rating given by a User")+theme_light()
```


\pagebreak

Next, we will going to plot the distribution of mean movie ratings given by users, to avoid outliers and high bias, only users with more than 100 movies rated are used. It can be seen from the plot below that most user give an average rating of between 3 and 4
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%',warning=FALSE,message=FALSE}
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(average = mean(rating)) %>%
  ggplot(aes(average)) + geom_histogram(bins = 25, color = "grey") +
  xlab("Average rating") + ylab("Number of users") +
  ggtitle("Average movie ratings given by users") +
  theme_light()
```


#3. Modelling with Least Square Error and Regularization
## Naive Models

We will first construct a naive model by giving the same predictions for all of the movies in the validation dataset which is the average of all the movie ratings. 
First we calculate the mean of the ratings given using the following code, it can be seen that the mean is 3.512465 which means that all of the future movie will be predicted to have a rating of 3.512465.
```{r, echo=TRUE}
mu <- mean(edx$rating)
mu
```

This average of the movie rating will also be used in further models as a baseline model. The following RMSE is obtained for the validation dataset
```{r, echo=TRUE}
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse
```

Then by using the following code, we will save the result of the RMSE obtained for this Naive Average Model so that later it can be used as a comparison for the other models.
```{r, echo=T, results='hide'}
rmse_results <- data_frame(method = "Naive Average movie rating model", RMSE = naive_rmse)
```
```{r, include=TRUE, echo=TRUE}
rmse_results %>% knitr::kable()
```


## Movie Effect Models

Now we will try to take into account the effect of movie in the model. since the it is logical that every movie will have its own unique rating, we will now give a movie effect parameter denoted by $b_i$ or b_i in the code. $b_i$ can be obtained by getting the average of subtracting every rating received in a movie by the mean which is the 3.15 in the previous naive model.
```{r, echo=T, results='hide', warning=FALSE, message=FALSE}
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
```

The plot that shows the distribution of $b_i$ generated can be seen below. It can be seen that most movies will have a computer $b_i$ with the value near to 0.
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
# Plot number of movies with the computed b_i
movie_avgs %>% ggplot(aes(b_i))+geom_histogram(bins=25,color="grey")+
  xlab('movie effect coefficient b_i')+ylab('Number of Movies')+
  ggtitle("Number of Movies For Every Computed b_i")+theme_light()
```

Now, Prediction are done by adding the mean with the b_i corresponding to the movieID in validation dataset. Then RMSE will also be calculated to indicate the performance of the model.
```{r, echo=TRUE}
predicted_ratings_1 <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings_1, validation$rating)
```


The following code is used to add the current model's RMSE in the table previously created. This same code will also be applied for the future model. From the result obtained, we can see that RMSE obtained decreased from the initial value of larger than 1 to less than 1 which is 0.944.
```{r, echo=TRUE}
rmse_results <- bind_rows(rmse_results,
data_frame(method="Movie effect model",  
RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()
```

## Movie and user effect model

Continuing with the previous idea, now we can see that besides movies, every users also have different tendencies on giving movie ratings. Some users like to give high ratings to any movies while other may be very objective and nitpicky in giving ratings. Due to this, in this model we are going to add an additional user effect denoted by $b_u$ or b_u to the previously movie effect model. First, we can see the distribution of the computed $b_u$. The plot constructed are based on users that has already rated more than 100 movies to prevent any outliers or bias in the plot. It can be seen below that the distribution of computed b_u is similar to normal distribution
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
edx %>% 
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
filter(n() >= 100) %>%
summarize(b_u = mean(rating - mu - b_i))%>%
ggplot(aes(b_u))+geom_histogram(bins=25,color="grey")+
xlab('User effect coefficient b_u')+ylab('Number of Movies')+
ggtitle("Number of Movies For Every Computed b_u")+theme_light()
```

Then we compute the value $b_u$ for all users
```{r, echo=T, results='hide', warning=FALSE, message=FALSE}
user_avgs <- edx %>%
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu - b_i))
```

Prediction are done by adding the mean with the $b_i$ corresponding to the movieID and $b_u$ corresponding to the userID in validation dataset
```{r, echo=TRUE}
predicted_ratings_2 <- validation%>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
mutate(pred = mu + b_i + b_u) %>%
pull(pred)
```

The following table shows that there is an improvement from the previous model since the RMSE obtained become better with the value 0.865
```{r, include=TRUE, echo=FALSE}
model_2_rmse <- RMSE(predicted_ratings_2, validation$rating)
rmse_results <- bind_rows(rmse_results,
data_frame(method="Movie and user effect model",  
RMSE = model_2_rmse))

rmse_results %>% knitr::kable()
```

## Regularized movie effect model 

Now we are going to see that there are bias caused by small amount of observation in a group, in this case small amount of ratings given to a movie and small amount of ratings done by a user would cause a bias and hence affecting the performance of the model. Due to this, we will add a regularization parameter that gives a big change (commonly known as penalty) if the amount of observation is smaller and will give small to no change as the number of observations increase indefinitely.

To observe this change, we will now add a regularization to the previously created movie effect model. The initial step is to create a set of values which will be the candidates for the regularization parameter more commonly known as lambda or $\lambda$. 
```{r, include=TRUE, echo=FALSE}
lambdas <- seq(0, 10, 0.25)
```

Then we will do a cross validation for each value of $\lambda$. first $b_i$ will be computed by using the now modified formula of least square error with the additional $\lambda$, then the predicted ratings for the validation dataset is also calculated with similar approach in previous models. The $\lambda$ value that will be chosen is the value that minimizes RMSE.

```{r, echo=T, results='hide', warning=FALSE, message=FALSE}
rmses_bi <- sapply(lambdas, function(l){
b_i <- edx %>% 
group_by(movieId) %>% 
summarize(b_i = sum(rating - mu)/(n()+l))

predicted_ratings<-validation%>%
left_join(b_i, by='movieId') %>% 
mutate(pred = mu + b_i) %>%
.$pred
return(RMSE(predicted_ratings, validation$rating))
})
```

using the plot below it can be seen that the lambda that returns the smallest RMSE is around 2.5 and when calculated exactly it will also produce the value 2.5
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
qplot(lambdas, rmses_bi,ylab="RMSE",xlab="Lambda",
main="RMSEs of Regularized Movie Effect Model")
```

```{r, include=FALSE, echo=FALSE}
lambdas[which.min(rmses_bi)]
```

Adding this results with the previous results shows that the RMSE obtained is not better then the movie and user effect model, however it has a small improvement from the initial movie effect model.
```{r, echo=T, results='hide', warning=FALSE, message=FALSE}
rmse_results <- bind_rows(rmse_results,
data_frame(method="Regularized movie effect model",  
RMSE = min(rmses_bi)))
```
```{r, include=TRUE, echo=FALSE}
rmse_results %>% knitr::kable()
```

## Regularized Movie and User Effect Model

Following the previous regularized model, next the movie and user effect model will also be regularized by first finding the best $\lambda$ value from the same set of candidates as before.
```{r, echo=T, results='hide', warning=FALSE, message=FALSE}
rmses_bi_bu <- sapply(lambdas, function(l){
b_i <- edx %>% 
group_by(movieId) %>%
summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx %>% 
left_join(b_i, by="movieId") %>% group_by(userId) %>%
summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- 
validation %>% 
left_join(b_i, by = "movieId") %>%
left_join(b_u, by = "userId") %>%
mutate(pred = mu + b_i + b_u) %>%
pull(pred)

return(RMSE(predicted_ratings, validation$rating))
})
```

The results in the plot below shows that the $\lambda$ that returns the smallest RMSE is aroung 5, and it can be checked that the exact amount for $\lambda$ is 5.25
```{r, fig.align='center', echo=FALSE, comment='', out.height = '40%'}
qplot(lambdas, rmses_bi_bu,ylab="RMSE",xlab="Lambda",
main="RMSEs of Regularized Movie and User Effect Model")  
```

```{r, include=TRUE, echo=FALSE}
lambdas[which.min(rmses_bi_bu)]

rmse_results <- bind_rows(rmse_results,
data_frame(method="Regularized movie and user effect model",  
RMSE = min(rmses_bi_bu)))
```

In the result in table below we can see that there is a slight improvement compared to the last model and user effect model.
```{r, include=TRUE, echo=FALSE}
rmse_results %>% knitr::kable()
```

# Results

The table below shows the final result of RMSE from different kind of models used to predict movie ratings
```{r, include=TRUE, echo=FALSE}
rmse_results %>% knitr::kable()
```
From the table, it is clear that as we increase the amount of effect used to explain the model, the RMSE will also decrease indicating a better performance. This is because there might be less error and the model having more capabilities to express the real observations. Then, we can conclude that the best model that can be used to predict movie ratings is the regularized movie and user effect with RMSE 0.8648170. 

However it is also advised to conduct a further observation whether the regularized movie and user effect do perform better than the normal movie and user effect. This is due to the relatively small difference between the RMSE of the two model (around 0.0005), which might not be that significant than 0. The significance of this difference needs to be checked by using statistical testing between two means.

# Appendix
```{r, include=TRUE, echo=FALSE}
print("Operating System:")
version
```