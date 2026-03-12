import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from src.data.load_data import load_data
from src.utils.utility import get_file_names

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        files = get_file_names('resources/anonymized_data')
        with ProcessPoolExecutor(max_workers=3) as executor:
            df_list = list(executor.map(load_data, files))

        combined_df = pd.concat(df_list, ignore_index=True)
        logging.info(combined_df.shape)
        logging.info(combined_df['source_file'].value_counts())
    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("DONE")

