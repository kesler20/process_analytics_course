import pyphi as phi
import pyphi_plots as pp
import pandas as pd
# Read the excel file
filename = r"C:\Users\CBE-User 05\protocol\process_analytics_course\Process Analytics Course Materials 2023\intro_datasets\19_MB Data\EX_MB RollerCompaction.xls"
excel_file = pd.ExcelFile(filename)

# Get all the sheet names
sheet_names = excel_file.sheet_names

psd  = pd.read_excel(filename,sheet_name=sheet_names[0]) 
hgl  = pd.read_excel(filename,sheet_name=sheet_names[1])
sbvstv = pd.read_excel(filename,sheet_name=sheet_names[2])
gerteis_mvs = pd.read_excel(filename,sheet_name=sheet_names[3])
gerteis_resp = pd.read_excel(filename,sheet_name=sheet_names[4])
hcp =  pd.read_excel(filename,sheet_name=sheet_names[5])

mbdata = {
    sheet_names[0] : psd
}
# mbpls_obj=phi.mbpls(mbdata,cqa_data,2)
# preds=phi.pls_pred(mbdata, mbpls_obj)

# pp.r2pv(mbpls_obj)
# pp.mb_r2pb(mbpls_obj)
# pp.mb_weights(mbpls_obj)
# pp.mb_vip(mbpls_obj)
