"""Billing through /billing/balance, /billing/usage-history and /billing/usage-analytics."""

import asyncio
import re
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import get_args

import httpx2
import pytest
from pydantic import ValidationError
from typing_extensions import assert_type

from llm_venice._endpoints import (
    UsageAnalyticsQuery,
    UsageHistoryQuery,
    json_operation,
    parse_envelope,
    parse_usage_csv,
)
from llm_venice.client import AsyncVeniceClient, VeniceClient
from llm_venice.errors import AuthenticationError, VeniceResponseError
from llm_venice.results import UsageHistoryCsv
from llm_venice.types import (
    BillingBalance,
    BillingBalances,
    BillingCurrency,
    UsageAnalytics,
    UsageBreakdown,
    UsageByDate,
    UsageByKey,
    UsageByModel,
    UsageEntry,
    UsageHistoryPage,
    UsageInferenceDetails,
)

API = "https://api.venice.ai/api/v1/billing"
FIXTURE = Path(__file__).parent / "fixtures/openapi-2026-11-22.json"
START = datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=2)))
END = datetime(2026, 11, 2, tzinfo=timezone.utc)
BALANCE = {
    "canConsume": True,
    "consumptionCurrency": "DIEM",
    "balances": {"diem": 9.5, "usd": None, "bundledCredits": None, "earnedCredits": 1},
    "diemEpochAllocation": 10,
}
ENTRY = {
    "amount": 0.02,
    "apiKeyId": None,
    "currency": "USD",
    "inferenceDetails": None,
    "notes": "API Inference",
    "pricePerUnitUsd": 0.02,
    "sku": "qwen-image-2-1-turbo-edit-inpaint",
    "timestamp": "2026-11-24T13:40:00.000Z",
    "units": 1,
}
PAGE = {"data": [ENTRY], "nextCursor": "next_page-1"}
ANALYTICS = {
    "lookback": "7d",
    "byDate": [{"date": "2026-11-24", "USD": 0.1, "DIEM": 0.5}],
    "byModel": [],
    "byModelDaily": [],
    "topModels": [],
    "byKey": [],
    "byKeyDaily": [],
    "topKeyNames": [],
}


def fixture_schema(path):
    resource = json.loads(FIXTURE.read_text())["resources"][f"GET {path}"]
    return resource


def response_schema(path):
    return fixture_schema(path)["response"]["content"]["application/json"]["schema"]


def invoke(is_async, method, **kwargs):
    if is_async:

        async def run():
            async with AsyncVeniceClient(key="test") as client:
                return await getattr(client, method)(**kwargs)

        return asyncio.run(run())
    with VeniceClient(key="test") as client:
        return getattr(client, method)(**kwargs)


def parse(name, body):
    response = httpx2.Response(200, json=body, request=httpx2.Request("GET", API))
    return parse_envelope(response, name, "Billing")


def test_types_match_dated_fixture():
    balance = response_schema("/billing/balance")
    assert set(BillingBalance.__annotations__) == set(balance["properties"])
    assert BillingBalance.__required_keys__ <= set(balance["required"])
    assert set(BillingBalances.__annotations__) == set(
        balance["properties"]["balances"]["properties"]
    )
    page = response_schema("/billing/usage-history")
    entry = page["properties"]["data"]["items"]
    assert set(UsageHistoryPage.__annotations__) == set(page["properties"])
    assert UsageHistoryPage.__required_keys__ <= set(page["required"])
    assert set(UsageEntry.__annotations__) == set(entry["properties"])
    assert UsageEntry.__required_keys__ <= set(entry["required"])
    assert set(UsageInferenceDetails.__annotations__) == set(
        entry["properties"]["inferenceDetails"]["properties"]
    )
    assert set(get_args(BillingCurrency)) == set(entry["properties"]["currency"]["enum"])
    analytics = response_schema("/billing/usage-analytics")
    properties = analytics["properties"]
    assert set(UsageAnalytics.__annotations__) == set(properties)
    assert UsageAnalytics.__required_keys__ <= set(analytics["required"])
    for name, kind in (("byDate", UsageByDate), ("byModel", UsageByModel), ("byKey", UsageByKey)):
        assert set(kind.__annotations__) == set(properties[name]["items"]["properties"])
    breakdown = properties["byModel"]["items"]["properties"]["breakdown"]["items"]
    assert set(UsageBreakdown.__annotations__) == set(breakdown["properties"])


def test_history_query_matches_dated_fixture():
    parameters = {
        parameter["name"]: parameter["schema"]
        for parameter in fixture_schema("/billing/usage-history")["parameters"]
    }
    first = UsageHistoryQuery(
        currency="EARNED_CREDITS", start_timestamp=START, end_timestamp=END, page_size=10
    ).query()
    assert first == {
        "currency": "EARNED_CREDITS",
        "startTimestamp": "2026-10-31T23:30:00.000Z",
        "endTimestamp": "2026-11-02T00:00:00.000Z",
        "pageSize": "10",
    }
    assert set(first) | {"cursor"} == set(parameters)
    assert set(get_args(BillingCurrency)) == set(parameters["currency"]["enum"])
    page_size = parameters["pageSize"]
    for value in (page_size["minimum"], page_size["maximum"]):
        assert UsageHistoryQuery(page_size=value).page_size == value
    for value in (page_size["minimum"] - 1, page_size["maximum"] + 1):
        with pytest.raises(ValidationError):
            UsageHistoryQuery(page_size=value)
    cursor = parameters["cursor"]
    assert UsageHistoryQuery(cursor="a" * cursor["maxLength"]).query() == {
        "cursor": "a" * cursor["maxLength"]
    }
    for value in ("a" * (cursor["maxLength"] + 1), "has space", "a+b"):
        with pytest.raises(ValidationError):
            UsageHistoryQuery(cursor=value)
    for name in ("startTimestamp", "endTimestamp"):
        assert len(first[name]) <= parameters[name]["maxLength"]


@pytest.mark.parametrize(
    "inputs,message",
    [
        ({"cursor": "c", "currency": "USD"}, "send it without filters"),
        ({"cursor": "c", "page_size": 100}, "send it without filters"),
        ({"start_timestamp": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"end_timestamp": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"start_timestamp": END, "end_timestamp": END}, "later than start_timestamp"),
        ({"currency": "VCU"}, "Input should be"),
        ({"page_size": "100"}, "valid integer"),
        ({"other": 1}, "Extra inputs"),
    ],
)
def test_history_query_rejects(inputs, message):
    with pytest.raises(ValidationError, match=message):
        UsageHistoryQuery.model_validate(inputs)


def test_analytics_query_matches_dated_fixture():
    parameters = {
        parameter["name"]: parameter["schema"]
        for parameter in fixture_schema("/billing/usage-analytics")["parameters"]
    }
    assert parameters["lookback"]["default"] == "7d"
    assert UsageAnalyticsQuery().query() == {}
    assert UsageAnalyticsQuery(lookback="90d").query() == {"lookback": "90d"}
    dates = UsageAnalyticsQuery(start_date=date(2026, 11, 1), end_date=date(2026, 11, 1)).query()
    assert dates == {"startDate": "2026-11-01", "endDate": "2026-11-01"}
    assert set(dates) | {"lookback"} == set(parameters)
    for name in ("startDate", "endDate"):
        pattern = parameters[name]["pattern"]
        assert re.fullmatch(pattern, dates[name])


@pytest.mark.parametrize(
    "inputs,message",
    [
        ({"lookback": "91d"}, "at most 90d"),
        ({"lookback": "0d"}, "String should match"),
        ({"lookback": "7"}, "String should match"),
        ({"start_date": date(2026, 11, 1)}, "given together"),
        ({"end_date": date(2026, 11, 1)}, "given together"),
        (
            {"lookback": "7d", "start_date": date(2026, 11, 1), "end_date": date(2026, 11, 2)},
            "not both",
        ),
        ({"start_date": date(2026, 11, 2), "end_date": date(2026, 11, 1)}, "not be before"),
    ],
)
def test_analytics_query_rejects(inputs, message):
    with pytest.raises(ValidationError, match=message):
        UsageAnalyticsQuery.model_validate(inputs)


def test_operations():
    assert json_operation("billing_balance", {}) == (
        "GET",
        "billing/balance",
        "Retrieving billing balance",
        {},
    )
    method, path, _, kwargs = json_operation("usage_history", {"cursor": "c"})
    assert (method, path, kwargs) == ("GET", "billing/usage-history", {"params": {"cursor": "c"}})
    method, path, _, kwargs = json_operation("usage_analytics", {"lookback": "30d"})
    assert (path, kwargs) == ("billing/usage-analytics", {"params": {"lookback": "30d"}})


def test_responses_tolerate_optional_drift():
    drifted = {"canConsume": False, "balances": {}, "consumptionCurrency": None, "future": 1}
    assert parse("billing_balance", drifted) == drifted
    last = {"data": [{**ENTRY, "notes": None, "units": "1"}], "nextCursor": None}
    assert parse("usage_history", last) == last
    assert parse("usage_analytics", {"lookback": "2026-11-01:2026-11-02"})["lookback"]


@pytest.mark.parametrize(
    "name,body,detail",
    [
        ("billing_balance", {"balances": {}}, "invalid BillingBalance: canConsume"),
        ("billing_balance", {"canConsume": True}, "invalid BillingBalance: balances"),
        (
            "billing_balance",
            {"canConsume": True, "balances": {"usd": "1"}},
            "invalid BillingBalance: balances",
        ),
        ("usage_history", {"data": []}, "nextCursor"),
        ("usage_history", {"data": [], "nextCursor": 1}, "nextCursor"),
        ("usage_history", {"nextCursor": None}, "data array"),
        ("usage_history", {"data": [{"amount": 1}], "nextCursor": None}, "UsageEntry"),
        ("usage_analytics", {"byDate": []}, "invalid UsageAnalytics: lookback"),
    ],
)
def test_malformed_responses(name, body, detail):
    with pytest.raises(VeniceResponseError, match=detail):
        parse(name, body)


def test_csv_pages():
    request = httpx2.Request("GET", API)
    csv = "timestamp,amount\n2026-11-24T13:40:00.000Z,0.02\n"
    page = httpx2.Response(
        200, text=csv, headers={"content-type": "text/csv", "x-next-cursor": "n-1"}, request=request
    )
    assert parse_usage_csv(page) == UsageHistoryCsv(csv, "n-1")
    last = httpx2.Response(200, text=csv, headers={"content-type": "text/csv"}, request=request)
    assert parse_usage_csv(last) == UsageHistoryCsv(csv, None)
    json_page = httpx2.Response(200, json=PAGE, request=request)
    with pytest.raises(VeniceResponseError, match="expected CSV but received JSON"):
        parse_usage_csv(json_page)
    bad = httpx2.Response(200, text=csv, headers={"x-next-cursor": "a b"}, request=request)
    with pytest.raises(VeniceResponseError, match="invalid x-next-cursor header"):
        parse_usage_csv(bad)


@pytest.mark.parametrize("is_async", [False, True])
def test_balance_and_analytics_clients(is_async, httpx2_mock):
    httpx2_mock.add_response(method="GET", url=f"{API}/balance", json=BALANCE)
    assert invoke(is_async, "get_billing_balance") == BALANCE
    httpx2_mock.add_response(
        method="GET",
        url=f"{API}/usage-analytics?startDate=2026-11-01&endDate=2026-11-07",
        json=ANALYTICS,
    )
    result = invoke(
        is_async, "get_usage_analytics", start_date=date(2026, 11, 1), end_date=date(2026, 11, 7)
    )
    assert result == ANALYTICS
    assert httpx2_mock.get_requests()[0].headers["authorization"] == "Bearer test"


@pytest.mark.parametrize("is_async", [False, True])
def test_history_clients(is_async, httpx2_mock):
    httpx2_mock.add_response(
        method="GET",
        url=f"{API}/usage-history?currency=USD&startTimestamp=2026-10-31T23%3A30%3A00.000Z",
        json=PAGE,
    )
    page = invoke(is_async, "list_usage_history", currency="USD", start_timestamp=START)
    assert page == PAGE
    httpx2_mock.add_response(
        method="GET",
        url=f"{API}/usage-history?cursor=next_page-1",
        text="a,b\n",
        headers={"content-type": "text/csv; charset=utf-8"},
    )
    csv = invoke(is_async, "export_usage_history", cursor=page["nextCursor"])
    assert csv == UsageHistoryCsv("a,b\n", None)
    json_request, csv_request = httpx2_mock.get_requests()
    assert csv_request.headers["accept"] == "text/csv"
    assert csv_request.headers["authorization"] == "Bearer test"
    assert json_request.headers.get("accept") != "text/csv"


@pytest.mark.parametrize("is_async", [False, True])
def test_validation_precedes_http_and_errors_map(is_async, httpx2_mock):
    with pytest.raises(ValidationError):
        invoke(is_async, "list_usage_history", cursor="c", currency="USD")
    with pytest.raises(ValidationError):
        invoke(is_async, "export_usage_history", page_size=1)
    with pytest.raises(ValidationError):
        invoke(is_async, "get_usage_analytics", lookback="100d")
    assert not httpx2_mock.get_requests()
    httpx2_mock.add_response(
        url=f"{API}/balance", status_code=401, json={"error": "Admin API key required"}
    )
    with pytest.raises(AuthenticationError, match="Admin API key required"):
        invoke(is_async, "get_billing_balance")


def test_result_types(httpx2_mock):
    httpx2_mock.add_response(url=f"{API}/balance", json=BALANCE)
    httpx2_mock.add_response(url=f"{API}/usage-history", json=PAGE)
    httpx2_mock.add_response(url=f"{API}/usage-history", text="a\n")
    httpx2_mock.add_response(url=f"{API}/usage-analytics", json=ANALYTICS)
    with VeniceClient(key="test") as client:
        assert_type(client.get_billing_balance(), BillingBalance)
        assert_type(client.list_usage_history(), UsageHistoryPage)
        assert_type(client.export_usage_history(), UsageHistoryCsv)
        assert_type(client.get_usage_analytics(), UsageAnalytics)
