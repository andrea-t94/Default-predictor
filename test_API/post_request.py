''' test of aws and local deployment with python requests modul'''
import requests

payload = {"account_amount_added_12_24m":"0",
 "account_days_in_dc_12_24m":"0",
 "account_days_in_rem_12_24m":"0",
 "account_days_in_term_12_24m":"0",
 "account_incoming_debt_vs_paid_0_24m":"0",
 "account_status":"1",
 "account_worst_status_0_3m":1,
 "account_worst_status_12_24m":"",
 "account_worst_status_3_6m":"1",
 "account_worst_status_6_12m":"",
 "age":"20",
 "avg_payment_span_0_12m":"12.69230769",
 "avg_payment_span_0_3m":"8.333333333",
 "merchant_category":"Dietary supplements",
 "merchant_group":"Health & Beauty",
 "has_paid":True,
 "max_paid_inv_0_12m":"31638",
 "max_paid_inv_0_24m":"31638",
 "name_in_email":"no_match",
 "num_active_div_by_paid_inv_0_12m":"0.153846154",
 "num_active_inv":"2",
 "num_arch_dc_0_12m":"0",
 "num_arch_dc_12_24m":"0",
 "num_arch_ok_0_12m":"13",
 "num_arch_ok_12_24m":"14",
 "num_arch_rem_0_12m":"0",
 "num_arch_written_off_0_12m":"0",
 "num_arch_written_off_12_24m":"0",
 "num_unpaid_bills":"2",
 "status_last_archived_0_24m":"1",
 "status_2nd_last_archived_0_24m":"1",
 "status_3rd_last_archived_0_24m":"1",
 "status_max_archived_0_6_months":"1",
 "status_max_archived_0_12_months":"1",
 "status_max_archived_0_24_months":"1",
 "recovery_debt":"0",
 "sum_capital_paid_account_0_12m":"0",
 "sum_capital_paid_account_12_24m":"0",
 "sum_paid_inv_0_12m":"178839",
 "time_hours":"9.653333333",
 "worst_status_active_inv":"1"
 }

url="http://localhost:5000/api/detection" #only after local deployment

url_aws="http://defaultpredictorat-env.eba-qpsnfdxf.eu-west-2.elasticbeanstalk.com//api/detection"
header = {"Content-type": "application/json"}
response_decoded_json = requests.post(url_aws, json=payload, headers=header)
print(response_decoded_json.text)