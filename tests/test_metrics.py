import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "monthly_income": [3000, 5000, 4000, 6000, 7000, 8000],
            "job_satisfaction": [1, 3, 2, 4, 3, 4],
            "attrition": ["Yes", "No", "Yes", "No", "No", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_no_leavers(sample_df):
    df = sample_df.copy()
    df["attrition"] = "No"
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers(sample_df):
    df = sample_df.copy()
    df["attrition"] = "Yes"
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    # Sales: 1 leaver / 2 = 50%, HR: 1 leaver / 2 = 50%, IT: 0 leavers / 2 = 0%
    result = attrition_by_department(sample_df)
    rates = result.set_index("department")["attrition_rate"]
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0
    assert rates["IT"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    # Yes (employees 1,2,5): 1 leaver / 3 = 33.33%
    # No  (employees 3,4,6): 1 leaver / 3 = 33.33%
    result = attrition_by_overtime(sample_df)
    rates = result.set_index("overtime")["attrition_rate"]
    assert rates["Yes"] == 33.33
    assert rates["No"] == 33.33


def test_attrition_by_overtime_leaver_counts(sample_df):
    result = attrition_by_overtime(sample_df)
    counts = result.set_index("overtime")["leavers"]
    assert counts["Yes"] == 1
    assert counts["No"] == 1


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    # Leavers:  employees 1 (3000) and 3 (4000) -> avg 3500
    # Stayers: employees 2 (5000), 4 (6000), 5 (7000), 6 (8000) -> avg 6500
    result = average_income_by_attrition(sample_df)
    values = result.set_index("attrition")["avg_monthly_income"]
    assert values["Yes"] == 3500.0
    assert values["No"] == 6500.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    levels = list(result["job_satisfaction"])
    assert levels == sorted(levels)


def test_satisfaction_summary_rates_use_group_size_not_total_leavers(sample_df):
    # Level 1: 1 employee, 1 leaver -> 100% (old bug gave 50% by dividing by total leavers)
    # Level 2: 1 employee, 1 leaver -> 100%
    # Level 3: 2 employees, 0 leavers -> 0%
    # Level 4: 2 employees, 0 leavers -> 0%
    result = satisfaction_summary(sample_df)
    rates = result.set_index("job_satisfaction")["attrition_rate"]
    assert rates[1] == 100.0
    assert rates[2] == 100.0
    assert rates[3] == 0.0
    assert rates[4] == 0.0
