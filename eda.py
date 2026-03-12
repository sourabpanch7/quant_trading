import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from src.data.load_data import load_data
from src.utils.utility import get_file_names
from src.eda.correlation_analysis import CorrelationCalculation

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        files = get_file_names('resources/anonymized_data')
        with ProcessPoolExecutor(max_workers=3) as executor:
            df_list = list(executor.map(load_data, files))

        combined_df = pd.concat(df_list, ignore_index=True)

        correlation_obj = CorrelationCalculation(df=combined_df,
                                                 img_path=r'/Users/sourabpanchanan/PycharmProjects/quant_trading/resources/outputs/eda_plots')
        correlation_obj.perform_eda()

        logging.info(correlation_obj.leaders.head(10))

        logging.info(correlation_obj.followers.head(10))

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("DONE")
