

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
   
    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
    transformation_obj = DataTransformation()
    train_arr, test_arr, _ = transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    trainer_obj = ModelTrainer()
    r2_score = trainer_obj.initiate_model_training(train_arr, test_arr)
    print(f"The R2 score of the best model is: {r2_score}")