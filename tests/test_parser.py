from parse_health_xml import filter_records


SAMPLE_RECORDS = [
    {
        "type": "HKQuantityTypeIdentifierRestingHeartRate",
        "value": "58",
        "unit": "count/min",
        "startDate": "2026-03-01 08:00:00 +0100",
        "endDate":   "2026-03-01 08:00:00 +0100",
    },
    {
        "type": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
        "value": "42.5",
        "unit": "ms",
        "startDate": "2026-03-01 08:00:00 +0100",
        "endDate":   "2026-03-01 08:00:00 +0100",
    },
    {
        "type": "HKQuantityTypeIdentifierRestingHeartRate",
        "value": "61",
        "unit": "count/min",
        "startDate": "2026-03-02 08:00:00 +0100",
        "endDate":   "2026-03-02 08:00:00 +0100",
    },
]


def test_filter_returns_only_matching_type():
    result = filter_records(SAMPLE_RECORDS, "HKQuantityTypeIdentifierRestingHeartRate")
    assert len(result) == 2
    assert all(r["type"] == "HKQuantityTypeIdentifierRestingHeartRate" for r in result)


def test_filter_returns_empty_for_unknown_type():
    result = filter_records(SAMPLE_RECORDS, "HKQuantityTypeIdentifierStepCount")
    assert result == []


def test_filter_returns_empty_for_empty_input():
    result = filter_records([], "HKQuantityTypeIdentifierRestingHeartRate")
    assert result == []


def test_filter_does_not_mutate_input():
    original_length = len(SAMPLE_RECORDS)
    filter_records(SAMPLE_RECORDS, "HKQuantityTypeIdentifierRestingHeartRate")
    assert len(SAMPLE_RECORDS) == original_length


def test_filter_single_match():
    result = filter_records(SAMPLE_RECORDS, "HKQuantityTypeIdentifierHeartRateVariabilitySDNN")
    assert len(result) == 1
    assert result[0]["value"] == "42.5"
