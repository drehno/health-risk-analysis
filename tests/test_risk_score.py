import pandas as pd
from risk_score import (
    assign_risk_level,
    compute_risk_score,
    HIGH_LOAD_3D_MINUTES,
    HRV_SUPPRESSION_MS,
    RESTING_HR_ELEVATION_BPM,
)


def make_row(**kwargs) -> pd.Series:
    """Returns a pd.Series with all scoring fields set to safe defaults."""
    defaults = {
        "sleep_hours": 7.5,
        "fatigue": 3,
        "soreness": 2,
        "resting_hr_diff_from_7d_mean": 0.0,
        "hrv_diff_from_7d_mean": 0.0,
        "workout_load_3d_sum": 30.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestAssignRiskLevel:
    def test_score_0_is_low(self):
        assert assign_risk_level(0) == "low"

    def test_score_2_is_low(self):
        assert assign_risk_level(2) == "low"

    def test_score_3_is_medium(self):
        assert assign_risk_level(3) == "medium"

    def test_score_5_is_medium(self):
        assert assign_risk_level(5) == "medium"

    def test_score_6_is_high(self):
        assert assign_risk_level(6) == "high"

    def test_score_10_is_high(self):
        assert assign_risk_level(10) == "high"


class TestComputeRiskScore:
    def test_healthy_day_scores_zero(self):
        assert compute_risk_score(make_row()) == 0

    def test_short_sleep_adds_two(self):
        assert compute_risk_score(make_row(sleep_hours=5.9)) == 2

    def test_sleep_exactly_six_does_not_trigger(self):
        assert compute_risk_score(make_row(sleep_hours=6.0)) == 0

    def test_high_fatigue_adds_two(self):
        assert compute_risk_score(make_row(fatigue=8)) == 2

    def test_fatigue_below_threshold_does_not_trigger(self):
        assert compute_risk_score(make_row(fatigue=7)) == 0

    def test_high_soreness_adds_two(self):
        assert compute_risk_score(make_row(soreness=8)) == 2

    def test_elevated_resting_hr_adds_one(self):
        assert compute_risk_score(
            make_row(resting_hr_diff_from_7d_mean=RESTING_HR_ELEVATION_BPM + 0.1)
        ) == 1

    def test_resting_hr_at_threshold_does_not_trigger(self):
        assert compute_risk_score(
            make_row(resting_hr_diff_from_7d_mean=RESTING_HR_ELEVATION_BPM)
        ) == 0

    def test_suppressed_hrv_adds_one(self):
        assert compute_risk_score(
            make_row(hrv_diff_from_7d_mean=-(HRV_SUPPRESSION_MS + 0.1))
        ) == 1

    def test_hrv_at_threshold_does_not_trigger(self):
        assert compute_risk_score(
            make_row(hrv_diff_from_7d_mean=-HRV_SUPPRESSION_MS)
        ) == 0

    def test_high_training_load_adds_two(self):
        assert compute_risk_score(
            make_row(workout_load_3d_sum=HIGH_LOAD_3D_MINUTES + 1)
        ) == 2

    def test_all_rules_fire_scores_ten(self):
        row = make_row(
            sleep_hours=4.0,
            fatigue=9,
            soreness=9,
            resting_hr_diff_from_7d_mean=RESTING_HR_ELEVATION_BPM + 1,
            hrv_diff_from_7d_mean=-(HRV_SUPPRESSION_MS + 1),
            workout_load_3d_sum=HIGH_LOAD_3D_MINUTES + 1,
        )
        assert compute_risk_score(row) == 10

    def test_nan_sleep_skips_rule(self):
        assert compute_risk_score(make_row(sleep_hours=float("nan"))) == 0

    def test_nan_resting_hr_deviation_skips_rule(self):
        assert compute_risk_score(
            make_row(resting_hr_diff_from_7d_mean=float("nan"))
        ) == 0

    def test_nan_hrv_deviation_skips_rule(self):
        assert compute_risk_score(
            make_row(hrv_diff_from_7d_mean=float("nan"))
        ) == 0
