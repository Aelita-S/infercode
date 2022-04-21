from infercode.client.infercode_trainer import InferCodeTrainer
from infercode.settings import DATASET_DIR

if __name__ == '__main__':
    input_data_path = DATASET_DIR / "OJ_raw_small"
    output_processed_data_path = DATASET_DIR / "OJ_raw_processed" / "OJ_raw_small.pkl"

    val_data_path = DATASET_DIR / "OJ_raw_val"
    val_processed_data_path = DATASET_DIR / "OJ_raw_processed" / "OJ_raw_val.pkl"
    infercode_trainer = InferCodeTrainer()
    infercode_trainer.process_data_sequence(input_data_path=input_data_path,
                                            output_processed_data_path=output_processed_data_path,
                                            val_data_path=val_data_path,
                                            val_processed_data_path=val_processed_data_path)

    infercode_trainer.train(workers=1)
    # infercode_trainer.test_train()
