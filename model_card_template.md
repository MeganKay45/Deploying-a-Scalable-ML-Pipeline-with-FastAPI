# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised machine learning classification model trained to predict whether an individual’s income is greater than $50,000 based on census data. The model was implemented using a RandomForestClassifier from scikit-learn. The input features include both categorical and numerical attributes from the census dataset, and the target label is salary.

## Intended Use
The intended use of this model is educational. It is designed to demonstrate an end-to-end machine learning pipeline, including data processing, model training, evaluation, and deployment with FastAPI. It should not be used to make real hiring, compensation, lending, or other high-stakes decisions about individuals.

## Training Data
The training data comes from the census dataset provided in the project starter repository. The dataset includes demographic and employment-related features such as workclass, education, marital-status, occupation, relationship, race, sex, native-country, age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week. The training split was created from this dataset using an 80/20 train-test split.

## Evaluation Data
The evaluation data consists of the held-out test portion of the same census dataset. This test set was not used during model training and was used to measure the model’s predictive performance after training.

## Metrics
The model was evaluated using precision, recall, and F1 score. On the test set, the model achieved a precision of 0.7419, a recall of 0.6384, and an F1 score of 0.6863. These metrics help measure the balance between correctly identifying positive cases and avoiding incorrect positive predictions.

## Ethical Considerations
This model uses demographic and socioeconomic features, some of which are sensitive, such as race and sex. Because of this, the model may reflect or amplify biases present in the underlying data. Predictions from this model could be unfair across different groups, so it is important not to use it in real-world decision-making without careful fairness analysis, bias testing, and human oversight.

## Caveats and Recommendations
This model was built for a classroom project and is limited by the quality, age, and representativeness of the census dataset. Its performance may not generalize well to other populations or time periods. The model should be considered a demonstration rather than a production-ready system. Future improvements could include hyperparameter tuning, fairness evaluation across demographic groups, more robust validation, and monitoring for performance drift after deployment.