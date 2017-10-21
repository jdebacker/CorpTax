
'''
------------------------------------------------------------------------
This script creates the tables for the Corp Tax paper but reading in
the pickled output, formatting it, and then saving TeX code for the tables
to a text file.

This py-file calls the following other file(s):

This py-file creates the following other file(s):

------------------------------------------------------------------------
'''

# import packages
import pickle
import os
import xlsxwriter

# read in pickles from model runs
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'OUTPUT'
output_dir = os.path.join(cur_path, output_fldr)
run_list = ['baseline', 'CIT', 'CFT']
eqm_list = ['PE', 'GE']
results = {}
for eq in eqm_list:
    for item in run_list:
        pkl_path = os.path.join(output_dir, eq, item, 'model_output.pkl')
        results[(eq, item)] = pickle.load(open(pkl_path, 'rb'))

# Data moments - enter here manually for now
cross_section_data = {'agg_IK': 0.095, 'agg_DE': 0.137, 'agg_SI': 0.130,
                      'agg_BV': 0.12, 'sd_IK': 0.156, 'sd_EK': 0.623,
                      'ac_IK': 0.596, 'ac_EK': 0.791,
                      'corr_BV_EK': -0.001}
regimes_data = {'frac_neg_debt': 0.319}
macro_data = {}
data_moments = {'cross_section': cross_section_data, 'macro': macro_data,
                'regimes': regimes_data}


# open Excel workbook
workbook = xlsxwriter.Workbook('CorpTaxTables.xlsx')

'''
Table 1: Moments, data vs model
'''
# get model moments unpacked
model_moments = results[('GE', 'baseline')]['moments']

# write to workbook
worksheet = workbook.add_worksheet('Moments')
worksheet.write(0, 0, 'Moment')
worksheet.write(0, 1, 'Data')
worksheet.write(0, 2, 'Model')
row_names = ['Investment rate', 'Aggregate dividends/earnings',
             'Aggregate new equity/investment',
             'Volatility of investment rate',
             'Autocorrelation of investment rate',
             'Volatility of earnings/capital',
             'Autocorrelation of earnings/capital',
             'Aggregate leverage ratio',
             'Frequency of negative debt',
             'Corr(Earnings, Leverage)']
moment_names = ['agg_IK', 'agg_DE', 'agg_SI', 'sd_IK', 'ac_IK', 'sd_EK',
                'ac_EK', 'agg_BV', 'frac_neg_debt', 'corr_BV_EK']
row = 1
for i in range(len(moment_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    if row_names[i] == 'Frequency of negative debt':
        worksheet.write(row, col, data_moments['regimes'][moment_names[i]])
    else:
        worksheet.write(row, col, data_moments['cross_section'][moment_names[i]])
    col += 1
    if row_names[i] == 'Frequency of negative debt':
        worksheet.write(row, col, model_moments['regimes'][moment_names[i]])
    else:
        worksheet.write(row, col, model_moments['cross_section'][moment_names[i]])
    row += 1

'''
Table 2: Baseline moments by financing regime
'''
# get model moments unpacked
model_regimes = results[('GE', 'baseline')]['moments']['regimes']

# write to workbook
worksheet = workbook.add_worksheet('Regimes')
worksheet.write(0, 0, 'Moment')
worksheet.write(0, 1, 'Equity Issuance')
worksheet.write(0, 2, 'Liquidity Constrained')
worksheet.write(0, 3, 'Dividend Distribution')
row_names = ['Share of firms', 'Share of capital', 'Share of investment',
             'Earnings/capital ratio', 'Investment rate', 'Average Q',
             'Leverage ratio', 'Frequency of debt > 0']
equity_names = ['frac_equity', 'share_K_equity', 'share_I_equity',
                'EK_equity', 'IK_equity', 'AvgQ_equity', 'BV_equity',
                'frac_equity_debt']
constrained_names = ['frac_constrained', 'share_K_constrained',
                     'share_I_constrained', 'EK_constrained',
                     'IK_constrained', 'AvgQ_constrained',
                     'BV_constrained', 'frac_constrained_debt']
div_names = ['frac_div', 'share_K_div', 'share_I_div', 'EK_div', 'IK_div',
             'AvgQ_div', 'BV_div', 'frac_div_debt']
row = 1
for i in range(len(row_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    worksheet.write(row, col, model_regimes[equity_names[i]])
    col += 1
    worksheet.write(row, col, model_regimes[constrained_names[i]])
    col += 1
    worksheet.write(row, col, model_regimes[div_names[i]])
    row += 1

'''
Table 3: Changes in macro aggregates in reforms
'''
# get model moments unpacked
base_macro = results[('GE', 'baseline')]['moments']['macro']
cit_macro = results[('GE', 'CIT')]['moments']['macro']
cft_macro = results[('GE', 'CFT')]['moments']['macro']

# write to workbook
worksheet = workbook.add_worksheet('Macro Changes')
worksheet.write(0, 0, 'Macro Aggregate/Price')
worksheet.write(0, 1, '20% Corporate Income Tax')
worksheet.write(0, 2, '20% Cash Flow Tax')
row_names = ['GDP', 'Investment', 'Consumption', 'Labor',
             'Accounting Profits', 'Average Q', 'Total Taxes',
             'Corporate Income Taxes']
moment_names = ['agg_Y', 'agg_I', 'agg_C', 'agg_L_d', 'agg_E', 'AvgQ',
                'total_taxes', 'agg_CIT']

row = 1
for i in range(len(row_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    worksheet.write(row, col, 100 * (cit_macro[moment_names[i]] -
                                     base_macro[moment_names[i]]) /
                    base_macro[moment_names[i]])
    col += 1
    worksheet.write(row, col, 100 * (cft_macro[moment_names[i]] -
                                     base_macro[moment_names[i]]) /
                    base_macro[moment_names[i]])
    row += 1

'''
Table 4: Changes in financial policies in reforms
'''
# get model moments unpacked
base_macro = results[('GE', 'baseline')]['moments']['macro']
cit_macro = results[('GE', 'CIT')]['moments']['macro']
cft_macro = results[('GE', 'CFT')]['moments']['macro']
base_cs = results[('GE', 'baseline')]['moments']['cross_section']
cit_cs = results[('GE', 'CIT')]['moments']['cross_section']
cft_cs = results[('GE', 'CFT')]['moments']['cross_section']

# write to workbook
worksheet = workbook.add_worksheet('Financial Policy Changes')
worksheet.write(0, 0, '')
worksheet.write(0, 1, '20% Corporate Income Tax')
worksheet.write(0, 2, '20% Cash Flow Tax')
row_names = ['Dividends', 'New Equity', 'Corporate Debt', 'Payout Ratio',
             'New equity/investment', 'Leverage ratio']
moment_names = ['agg_D', 'agg_S', 'agg_B', 'agg_DE', 'agg_SI', 'agg_BV']

row = 1
for i in range(len(row_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cit_macro[moment_names[i]] -
                                         base_macro[moment_names[i]]) /
                        base_macro[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cit_cs[moment_names[i]] -
                                         base_cs[moment_names[i]]) /
                        base_cs[moment_names[i]])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cft_macro[moment_names[i]] -
                                         base_macro[moment_names[i]]) /
                        base_macro[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cft_cs[moment_names[i]] -
                                         base_cs[moment_names[i]]) /
                        base_cs[moment_names[i]])
    row += 1

'''
Table 5: Changes in macro aggregates - comparing PE to GE results
'''
# get model moments unpacked
base_macro_pe = results[('PE', 'baseline')]['moments']['macro']
cit_macro_pe = results[('PE', 'CIT')]['moments']['macro']
cft_macro_pe = results[('PE', 'CFT')]['moments']['macro']

# write to workbook
worksheet = workbook.add_worksheet('Macro PE vs GE')
worksheet.write(1, 0, 'Macro Aggregate/Price')
worksheet.merge_range('B1:C1', '20% Corporate Income Tax')
worksheet.merge_range('D1:E1', '20% Cash Flow Tax')
worksheet.write(1, 1, 'PE')
worksheet.write(1, 2, 'GE')
worksheet.write(1, 3, 'PE')
worksheet.write(1, 4, 'GE')
row_names = ['GDP', 'Investment', 'Consumption', 'Labor',
             'Accounting Profits', 'Average Q', 'Total Taxes',
             'Corporate Income Taxes']
moment_names = ['agg_Y', 'agg_I', 'agg_C', 'agg_L_d', 'agg_E', 'AvgQ',
                'total_taxes', 'agg_CIT']

row = 2
for i in range(len(row_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    worksheet.write(row, col, 100 * (cit_macro_pe[moment_names[i]] -
                                     base_macro_pe[moment_names[i]]) /
                    base_macro_pe[moment_names[i]])
    col += 1
    worksheet.write(row, col, 100 * (cit_macro[moment_names[i]] -
                                     base_macro[moment_names[i]]) /
                    base_macro[moment_names[i]])
    col += 1
    worksheet.write(row, col, 100 * (cft_macro_pe[moment_names[i]] -
                                     base_macro_pe[moment_names[i]]) /
                    base_macro_pe[moment_names[i]])
    col += 1
    worksheet.write(row, col, 100 * (cft_macro[moment_names[i]] -
                                     base_macro[moment_names[i]]) /
                    base_macro[moment_names[i]])
    row += 1

'''
Table 6: Changes in financial policies - comparing PE and GE results
'''
# get model moments unpacked
base_macro_pe = results[('PE', 'baseline')]['moments']['macro']
cit_macro_pe = results[('PE', 'CIT')]['moments']['macro']
cft_macro_pe = results[('PE', 'CFT')]['moments']['macro']
base_cs_pe = results[('PE', 'baseline')]['moments']['cross_section']
cit_cs_pe = results[('PE', 'CIT')]['moments']['cross_section']
cft_cs_pe = results[('PE', 'CFT')]['moments']['cross_section']

# write to workbook
worksheet = workbook.add_worksheet('Financial Policies - PE vs GE')
worksheet.write(1, 0, '')
worksheet.merge_range('B1:C1', '20% Corporate Income Tax')
worksheet.merge_range('D1:E1', '20% Cash Flow Tax')
worksheet.write(1, 1, 'PE')
worksheet.write(1, 2, 'GE')
worksheet.write(1, 3, 'PE')
worksheet.write(1, 4, 'GE')
row_names = ['Dividends', 'New Equity', 'Corporate Debt', 'Payout Ratio',
             'New equity/investment', 'Leverage ratio']
moment_names = ['agg_D', 'agg_S', 'agg_B', 'agg_DE', 'agg_SI', 'agg_BV']

row = 2
for i in range(len(row_names)):
    col = 0
    worksheet.write(row, col, row_names[i])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cit_macro_pe[moment_names[i]] -
                                         base_macro_pe[moment_names[i]]) /
                        base_macro_pe[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cit_cs_pe[moment_names[i]] -
                                         base_cs_pe[moment_names[i]]) /
                        base_cs_pe[moment_names[i]])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cit_macro[moment_names[i]] -
                                         base_macro[moment_names[i]]) /
                        base_macro[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cit_cs[moment_names[i]] -
                                         base_cs[moment_names[i]]) /
                        base_cs[moment_names[i]])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cft_macro_pe[moment_names[i]] -
                                         base_macro_pe[moment_names[i]]) /
                        base_macro_pe[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cft_cs_pe[moment_names[i]] -
                                         base_cs_pe[moment_names[i]]) /
                        base_cs_pe[moment_names[i]])
    col += 1
    if i < 3:
        worksheet.write(row, col, 100 * (cft_macro[moment_names[i]] -
                                         base_macro[moment_names[i]]) /
                        base_macro[moment_names[i]])
    else:
        worksheet.write(row, col, 100 * (cft_cs[moment_names[i]] -
                                         base_cs[moment_names[i]]) /
                        base_cs[moment_names[i]])
    row += 1

workbook.close()
