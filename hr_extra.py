import json
import os

import pandas as pd
from loguru import logger


class HrResult(object):
    def __init__(self):
        logger.info("Start to initialize HrResult class.")
        print('a')
#        prefix = "./"
#        data_path = os.path.join(prefix, "data/")
#        config_path = os.path.join(prefix, "config/")
#        self.matching_table = pd.ExcelFile(data_path + "matching_table.xlsx", engine='openpyxl')
#        with open(config_path + "filter_switch.json", "r") as fg:
#            self.hr_switch = json.load(fg)
        # with open(config_path + "version.json", "r") as vj:
        #     self.version = json.load(vj)
        logger.info("HrResult Class initialized.")

#    def get_refer_table(self, refer_name, grade):
#        sheet_name = refer_name
#        switch = self.hr_switch[sheet_name]
#        refer_df = self.matching_table.parse(sheet_name)
#        if switch == "yes":
#            # filter by grade
#            filter_result = refer_df[refer_df.Grade == grade]
#            refer_df = filter_result.drop(["Grade"], axis=1)
#        extra_result = json.loads(refer_df.to_json(orient="records"))
#        return extra_result

    def hr_response(self, result_df, grade):
        logger.info("Formatting output response...")
        whole_result = dict()
        for idx, row in result_df.iterrows():
            pair_dict = dict()
            pair_dict["Question"] = row["QUESTION"]
            pair_dict["Answer"] = row["ANSWER"]
            pair_dict["Image"] = row["IMAGE"]
            pair_dict["SimScore"] = row["SimScore"]
            # if row["REFER"] != "EMPTY":
            #     result = self.get_refer_table(row["REFER"], grade)
            #     extra_result = dict()
            #     extra_result["Excel"] = result
            #     pair_dict.update(extra_result)
            # idx = "Q" + str(idx)
            whole_result[idx] = pair_dict
        # to ensure the result is json format
        # result_json = json.dumps(whole_result)
        return whole_result
