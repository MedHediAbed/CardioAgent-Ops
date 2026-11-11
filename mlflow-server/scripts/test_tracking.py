
import mlflow

# 1. Connexion au serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Création ou récupération de l'Experiment
experiment_name = "CardioAgent-Tracking-Test"

mlflow.set_experiment(experiment_name)

# 3. Création d'un Run
with mlflow.start_run(run_name="first-test"):

    # Enregistrement des Parameters
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # Enregistrement des Metrics (valeurs fictives de démonstration)
    mlflow.log_metric("accuracy", 0.90)
    mlflow.log_metric("loss", 0.15)

    print("Experiment et Run enregistrés avec succès !")
