import pandas as pd

from predict_single_pair_labelFusion import build_summary_row


def test_summary_row_uses_param_strings_not_epoch_fields():
    out = pd.DataFrame({
        "true": [0, 1, 1],
        "llm_pred": [0, 1, 0],
        "roberta_pred": [0, 1, 1],
        "fusion_pred": [0, 0, 1],
    })

    row = build_summary_row("demo", out, 12.4, {"num_epochs": 12}, {"num_epochs": 7})

    assert "roberta_epochs" not in row
    assert "fusion_epochs" not in row
    assert "roberta_params" in row
    assert "fusion_params" in row
    assert "roberta_f1" in row
    assert "fusion_f1" in row
