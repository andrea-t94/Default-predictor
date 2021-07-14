#list of all the features used
FEATURES = ["account_amount_added_12_24m","account_days_in_dc_12_24m","account_days_in_rem_12_24m",
            "account_days_in_term_12_24m","account_incoming_debt_vs_paid_0_24m","account_status",
            "account_worst_status_0_3m","account_worst_status_12_24m","account_worst_status_3_6m",
            "account_worst_status_6_12m","age","avg_payment_span_0_12m","avg_payment_span_0_3m",
            "merchant_category","merchant_group","has_paid","max_paid_inv_0_12m","max_paid_inv_0_24m",
            "name_in_email","num_active_div_by_paid_inv_0_12m","num_active_inv","num_arch_dc_0_12m",
            "num_arch_dc_12_24m","num_arch_ok_0_12m","num_arch_ok_12_24m","num_arch_rem_0_12m",
            "num_arch_written_off_0_12m","num_arch_written_off_12_24m","num_unpaid_bills",
            "status_last_archived_0_24m","status_2nd_last_archived_0_24m","status_3rd_last_archived_0_24m",
            "status_max_archived_0_6_months","status_max_archived_0_12_months","status_max_archived_0_24_months",
            "recovery_debt","sum_capital_paid_account_0_12m","sum_capital_paid_account_12_24m",
            "sum_paid_inv_0_12m","time_hours", "worst_status_active_inv"]

#dictionary about how to pre-process the missing-value for each feature, based on analysis on those
missing_val_config = {
    #features missing in dataset
    "account_days_in_dc_12_24m": "mean",
    "account_days_in_rem_12_24m": "mean",
    "account_days_in_term_12_24m": "mean",
    "account_incoming_debt_vs_paid_0_24m": "mean",
    "account_status": "zero",
    "account_worst_status_0_3m": "zero",
    "account_worst_status_12_24m": "zero",
    "account_worst_status_3_6m": "zero",
    "account_worst_status_6_12m": "zero",
    "avg_payment_span_0_12m": "mean",
    "avg_payment_span_0_3m": "mean",
    "num_active_div_by_paid_inv_0_12m": "mean",
    "num_arch_written_off_0_12m": "zero",
    "num_arch_written_off_12_24m": "zero",
    "worst_status_active_inv": "zero",
   #features not missing in dataset
    "account_amount_added_12_24m": "mean",
    "age": "mean",
    "max_paid_inv_0_12m": "mean",
    "max_paid_inv_0_24m": "mean",
    "num_active_inv": "mean",
    "num_arch_dc_0_12m": "mean",
    "num_arch_dc_12_24m": "mean",
    "num_arch_ok_0_12m": "mean",
    "num_arch_ok_12_24m": "mean",
    "num_arch_rem_0_12m": "mean",
    "num_unpaid_bills": "mean",
    "status_last_archived_0_24m": "zero",
    "status_2nd_last_archived_0_24m": "zero",
    "status_3rd_last_archived_0_24m": "zero",
    "status_max_archived_0_6_months": "zero",
    "status_max_archived_0_12_months": "zero",
    "status_max_archived_0_24_months": "zero",
    "recovery_debt": "mean",
    "sum_capital_paid_account_0_12m": "mean",
    "sum_capital_paid_account_12_24m":"mean",
    "sum_paid_inv_0_12m":"mean",
    "time_hours":"mean"
}


#dictionary about how to model categorical values
#those cannot be replaced if nan
categorical_config = {
     "merchant_category": "categorical",
     "merchant_group": "categorical",
     "has_paid": "bool",
     "name_in_email": "categorical"
}
