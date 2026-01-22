# DATA 612 - Recommender Systems

Coursework for the CUNY SPS Master's course on recommendation systems. The course covered topics including sparse matrix representation, sparse matrix opperations, dealing with cold start problems as well as lack of negative feedback, and the use of neural networks in recommender systems. Particular emphasis was placed on the unique issues present when attempting to evaluate recommender systems, with traditional error metrics often falling short. Projects of note are:

1. [The Final Project](assignments/Final/Tillmawitz_data_612_final.ipynb) - Training Recommenders on Amazon Reviews

    A project comparing the performance of Matrix Factorization, Neural Collaborative Filtering, and Two Tower models when generating product recommendations. Project highlights include visualizations and explanations of the different model architectures, explanation of loss function (Bayesian Personalize Ranking) and evaluation metric (Normalized Discounted Cumulative Gain) choice, and comparison of explicit negative sampling during training on model performance.

2. [Project 3](assignments/Tillmawitz_project_3.ipynb) - Implementing Stochastic Gradient Descent for Model Training

    This project consists of a simple matrix factorization recommender model trained using manually implemented Stochastic Gradient Descent as a learning exercise.

3. [Project 4](assignments/Tillmawitz_project_4.ipynb) - Experimenting With Diversity in Recommendations

    Two recommender models are developed, an Alternating Least Squares and an Item Based Collaborative Filtering model, and trained on the Netflix prize dataset. Movies were clustered and models were either required to span a minimum number of clusters when generating recommendations or not. The difference in recommendations was then analyzed with a focus on business implications.
