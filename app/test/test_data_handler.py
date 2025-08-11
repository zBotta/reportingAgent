
import pandas as pd
import os
import shutil
from projectSetup import Setup
from conf.projectConfig import Config as cf
from mods.dataHandler import DataHandler, Report
from mods.apiReportGenerator import ApiReport, ApiReports

def test_import_excel():
    dh = DataHandler()
    df = dh.import_reports()
    assert df.columns.to_list() == cf.DATA.DF_COLUMNS

def test_export_excel():
    dh = DataHandler()
    # Create a dataframe
    d1 = {"a": 1, "b": 2, "c": 44, "d": 551}
    d2 = {"a": 4, "b": 31, "c": 66, "d": 666}
    df = pd.DataFrame([d1,d2])
    app_folder_dest = "test/test_export"
    filename_prefix = "test-export"
    folder_path = os.path.join(cf.APP_PATH, app_folder_dest)
    if os.path.isdir(folder_path): # delete folder with previous tests files
        shutil.rmtree(folder_path) 
    # Export it to excel
    dh.export_df_to_excel(df=df,
                          xlsx_file_name=filename_prefix, 
                          app_folder_destination=app_folder_dest)
    
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames
    
def test_api_export():
    dh = DataHandler()
    # Create a single report with the structured outputs
    api_rep = ApiReport(
        report_name='Evening Collision Near Lyon on N346 Route', 
        what='a car accident', 
        when='Wednesday at 7:30 pm', 
        where='near Lyon on the N346 route', 
        who='two drivers returning home after work', 
        how='Vehicle A, a motorcycle, was riding in the left lane speeding to overtake traffic while Vehicle B, a delivery van ahead, slowed abruptly due to road construction ahead', 
        why='Vehicle A’s rider failed to anticipate the sudden deceleration of Vehicle B, causing a rear-end collision', 
        contingency_actions='Traffic officers quickly closed the affected lane to assist the injured motorcyclist and detour traffic; emergency medical teams arrived within minutes; authorities collected CCTV footage from nearby cameras and eyewitness accounts to determine fault', 
        report='On Wednesday at 7:30 pm, during rush hour near Lyon on the N346 route, a traffic accident occurred involving a motorcycle (Vehicle A) and a delivery van (Vehicle B). As the van slowed unexpectedly because of roadworks, the motorcyclist behind failed to reduce speed, resulting in a rear-end collision. The motorcyclist sustained minor injuries and was treated on-site by emergency responders. The incident caused temporary lane closures and substantial traffic delays. Investigators gathered CCTV footage and eyewitness statements to clarify the circumstances. This event highlighted the dangers of inattentiveness under changing road conditions, emphasizing the need for caution during construction zones.'
        )
    report_data = ApiReports(reports=[api_rep])

    app_folder_dest = cf.API.API_GEN_REPORTS_F
    filename_prefix = "test-api-export"
    folder_path = os.path.join(cf.APP_PATH, app_folder_dest)
    if os.path.isdir(folder_path): # delete folder with previous tests files
        shutil.rmtree(folder_path) 
    dh.export_to_excel_from_api_response(report_data=report_data, 
                                         model_name="gpt2", 
                                         filename=filename_prefix)
    # Check the file creation, check the string prefix in the filename
    for filenames in os.listdir(folder_path):
        assert filename_prefix in filenames