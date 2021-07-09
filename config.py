#dictionary about how to pre-process the missing-value for each feature, based on analysis on those

missing_val_config = {
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
    "worst_status_active_inv": "zero"
}

#dictionary about how to model categorical values
categorical_config = {
     "merchant_category": "categorical",
     "merchant_group": "categorical",
     "has_paid": "bool",
     "name_in_email": "categorical"
}

#model structure configuration
model_config = dict(
    model="baseline-default-predictor",
    dense=16,
    dropout=0.2,
    batch_size=256,
    epochs=5,
    learning_rate=1e-4,
    class_weight={0: 0.5, 1: 300} #class weight dictionary, given an unbalacend dataset  88688:1289 = 100:1.43
)

"""
from tensowrflow documentation
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
"""