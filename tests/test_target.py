import pytest
import pandas as pd
from src.target import build_target

def make_repayments(rows: list[dict]) -> pd.DataFrame:
    """
    Each dict needs: application_id, emi_number, due_date, paid_date, amount_paid
    paid_date = None means unpaid
    """
    return pd.DataFrame(rows)

AS_OF = "2026-08-31"

def test_clean_good_loan():
    """All 12 EMIs paid on time — should be good and fully observed"""
    rows = [
        {"application_id": "L001", "emi_number": i,
         "due_date": f"2025-{i:02d}-01", #formats the integer i as a 2-digit number with a leading zero if needed
         "paid_date": f"2025-{i:02d}-05",   # paid 4 days after due
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L001"].iloc[0]
    assert row["default_90dpd_12m"] == 0
    assert row["is_fully_observed"] == True


def test_bad_loan_90_days_late():
    """One EMI paid exactly 90 days late — should trigger bad"""
    rows = [
        {"application_id": "L002", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    # Override EMI 3 to be exactly 90 days late
    rows[2]["paid_date"] = "2025-05-31"   # due 2025-03-01, paid 90 days later
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L002"].iloc[0]
    assert row["default_90dpd_12m"] == 1
    assert row["is_fully_observed"] == True


def test_bad_loan_89_days_late():
    """89 days late is NOT bad — boundary condition"""
    rows = [
        {"application_id": "L003", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    rows[2]["paid_date"] = "2025-05-29"   # due 2025-03-01, paid 89 days later
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L003"].iloc[0]
    assert row["default_90dpd_12m"] == 0


def test_unpaid_overdue_90_days():
    """EMI still unpaid and 95 days past due — should be bad"""
    rows = [
        {"application_id": "L004", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    # EMI 5 unpaid, due date is 97 days before as_of
    rows[4]["due_date"] = "2026-05-26"    # 97 days before 2026-08-31
    rows[4]["paid_date"] = None
    rows[4]["amount_paid"] = 0
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L004"].iloc[0]
    assert row["default_90dpd_12m"] == 1


def test_unpaid_not_yet_overdue():
    """EMI unpaid but only 30 days past due — not bad yet"""
    rows = [
        {"application_id": "L005", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    rows[4]["due_date"] = "2026-08-01"    # 30 days before as_of
    rows[4]["paid_date"] = None
    rows[4]["amount_paid"] = 0
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L005"].iloc[0]
    assert row["default_90dpd_12m"] == 0


def test_partial_payment_treated_as_unpaid():
    """Partial payment with no paid_date — treated as unpaid, DPD from as_of"""
    rows = [
        {"application_id": "L006", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    # EMI 3 partially paid but paid_date is null — 97 days overdue
    rows[2]["due_date"] = "2026-05-26"
    rows[2]["paid_date"] = None
    rows[2]["amount_paid"] = 2000         # partial — but paid_date still null
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L006"].iloc[0]
    assert row["default_90dpd_12m"] == 1  # still bad, paid_date drives the logic


def test_fewer_than_12_emis_due():
    """Loan only has 8 EMIs due by as_of — not fully observed"""
    rows = [
        {"application_id": "L007", "emi_number": i,
         "due_date": f"2026-{i:02d}-01",  # all in 2026, only ~8 fall before Aug 31
         "paid_date": f"2026-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L007"].iloc[0]
    assert row["is_fully_observed"] == False


def test_late_then_regular():
    """EMI 2 paid 95 days late, rest fine — loan is still bad"""
    rows = [
        {"application_id": "L008", "emi_number": i,
         "due_date": f"2025-{i:02d}-01",
         "paid_date": f"2025-{i:02d}-05",
         "amount_paid": 5000}
        for i in range(1, 13)
    ]
    rows[1]["paid_date"] = "2025-06-04"   # due 2025-02-01, paid 95 days later
    result = build_target(make_repayments(rows), AS_OF)
    row = result[result["application_id"] == "L008"].iloc[0]
    assert row["default_90dpd_12m"] == 1