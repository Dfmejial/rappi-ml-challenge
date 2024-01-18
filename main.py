import logging

from data.reader import TitanicDataReader
from data.pre_processer import TitanicDataPreprocessor
from model.train import TitanicModelTrainer

logger = logging.getLogger(__name__)

class RappiChallenge:
    """
    A class representing the Rappi Challenge workflow.

    Methods:
    - run_challenge: Executes the Rappi Challenge workflow, including data reading, preprocessing, model training, and evaluation.
    """

    def run_challenge(self) -> str:
        """
        Executes the Rappi Challenge workflow.

        Returns:
        - str: The path to the saved trained model.
        """

        # Step 1: Read and preprocess data
        reader = TitanicDataReader()
        raw_data = reader.read_raw_data()
        raw_data = reader.remove_cols(raw_data)

        # Step 2: Split data and preprocess
        trainer = TitanicModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(raw_data)

        pre_processor = TitanicDataPreprocessor()
        X_train = pre_processor.apply_preprocessing(X_train)
        X_test = pre_processor.apply_preprocessing(X_test)

        # Step 3: Train and evaluate the model
        clf = trainer.train_model(X_train, y_train)
        precision, recall, f1 = trainer.evaluate_model(clf, X_test, y_test)

        logger.warning("Precision on test set: %s", precision)
        logger.warning("Recall on test set: %s", recall)
        logger.warning("F1 on test set: %s", f1)

        # Step 4: Save the trained model
        path = trainer.save_model(clf)

        logger.warning("Model saved on %s", path)

        return path
    

def main() -> None:
    """
    Runs the Rappi Challenge.
    """
    rappi_challenge = RappiChallenge()
    rappi_challenge.run_challenge()

if __name__ == "__main__":
    main()
