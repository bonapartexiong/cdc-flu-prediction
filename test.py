"""
Unit tests for cdc_flu_pipeline.py core functions.
Run with:  pytest test.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os

from cdc_flu_pipeline import (
    create_features,
    compute_prediction_features,
    build_model,
    train_and_evaluate,
    predict_future,
    check_alerts,
    should_send_alert,
    create_plot,
    save_artifacts,
    send_email,
    fetch_cdc_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal historical DataFrame covering 60+ weeks for lag_52."""
    np.random.seed(42)
    n = 70
    base = datetime(2023, 1, 2)
    dates = [base + timedelta(weeks=i) for i in range(n)]
    ilitotal = np.random.randint(800, 3000, size=n).astype(float)
    df = pd.DataFrame({
        'week_start': dates,
        'week': [(d.isocalendar()[1]) for d in dates],
        'ilitotal': ilitotal,
        'total_patients': ilitotal * 20,
        'percent_ili': ilitotal / (ilitotal * 20) * 100,
    })
    return df.sort_values('week_start').reset_index(drop=True)


# ---------------------------------------------------------------------------
# create_features
# ---------------------------------------------------------------------------

class TestCreateFeatures:
    def test_returns_dataframe_and_target_name(self, sample_df):
        df_feat, target = create_features(sample_df)
        assert target == 'ilitotal'
        assert isinstance(df_feat, pd.DataFrame)

    def test_drops_rows_with_nan_from_lags(self, sample_df):
        df_feat, _ = create_features(sample_df)
        # lag_52 drops 52 rows; the final DataFrame should be shorter
        assert len(df_feat) < len(sample_df)

    def test_includes_expected_feature_columns(self, sample_df):
        df_feat, _ = create_features(sample_df)
        assert 'lag_1' in df_feat.columns
        assert 'lag_8' in df_feat.columns
        assert 'roll_mean_4' in df_feat.columns
        assert 'roll_std_12' in df_feat.columns
        assert 'week_sin' in df_feat.columns
        assert 'week_cos' in df_feat.columns
        assert 'lag_52' in df_feat.columns
        assert 'trend_4w' in df_feat.columns

    def test_no_data_leakage_lag1(self, sample_df):
        """lag_1 at row i must equal ilitotal at row i-1, not row i."""
        df_feat, _ = create_features(sample_df)
        # create_features drops NaN rows so its index is not 0..N-1;
        # iterate by df_feat's own positional order.
        for pos in range(1, len(df_feat)):
            row_label = df_feat.index[pos]
            prev_label = df_feat.index[pos - 1]
            expected = sample_df.loc[prev_label, 'ilitotal']
            assert df_feat.loc[row_label, 'lag_1'] == expected


# ---------------------------------------------------------------------------
# compute_prediction_features
# ---------------------------------------------------------------------------

class TestComputePredictionFeatures:
    def test_computes_all_features(self, sample_df):
        last_row = sample_df.iloc[-1].copy()
        feat = compute_prediction_features(last_row, sample_df)
        expected_keys = [
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
            'roll_mean_2', 'roll_mean_4', 'roll_mean_8', 'roll_mean_12',
            'roll_std_2', 'roll_std_4', 'roll_std_8', 'roll_std_12',
            'week_sin', 'week_cos', 'lag_52', 'trend_4w',
        ]
        for k in expected_keys:
            assert k in feat, f"Missing key: {k}"

    def test_lag1_matches_last_known_value(self, sample_df):
        feat = compute_prediction_features(sample_df.iloc[-1], sample_df)
        assert feat['lag_1'] == sample_df['ilitotal'].iloc[-1]

    def test_seasonal_features_in_range(self, sample_df):
        feat = compute_prediction_features(sample_df.iloc[-1], sample_df)
        assert -1.1 <= feat['week_sin'] <= 1.1
        assert -1.1 <= feat['week_cos'] <= 1.1


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_returns_lightgbm_regressor(self):
        model = build_model()
        from lightgbm import LGBMRegressor
        assert isinstance(model, LGBMRegressor)

    def test_num_leaves_does_not_exceed_2_pow_max_depth(self):
        model = build_model()
        max_leaves = 2 ** model.max_depth
        assert model.num_leaves <= max_leaves, (
            f"num_leaves={model.num_leaves} > 2^max_depth={max_leaves}"
        )


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    def test_returns_model_features_and_metrics(self, sample_df):
        model, feature_cols, val_rmse, val_r2 = train_and_evaluate(sample_df)
        assert hasattr(model, 'predict')
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert val_rmse > 0
        assert -1.0 <= val_r2 <= 1.0

    def test_feature_cols_exclude_target_and_ids(self, sample_df):
        _, feature_cols, _, _ = train_and_evaluate(sample_df)
        excluded = {'week_start', 'week', 'ilitotal', 'total_patients', 'percent_ili'}
        for col in excluded:
            assert col not in feature_cols, f"{col} should not be in feature_cols"

    def test_chronological_split_is_honest(self, sample_df):
        """Validation R² should not be 1.0 (would indicate data leakage)."""
        _, _, val_rmse, val_r2 = train_and_evaluate(sample_df)
        assert val_r2 < 0.999, f"Validation R² is suspiciously high: {val_r2}"


# ---------------------------------------------------------------------------
# predict_future
# ---------------------------------------------------------------------------

class TestPredictFuture:
    def test_returns_dataframe_with_expected_columns(self, sample_df):
        model, feature_cols, _, _ = train_and_evaluate(sample_df)
        pred = predict_future(model, sample_df, feature_cols)
        assert isinstance(pred, pd.DataFrame)
        assert 'predicted_ili' in pred.columns
        assert 'ci_lower' in pred.columns
        assert 'ci_upper' in pred.columns

    def test_prediction_is_non_negative(self, sample_df):
        model, feature_cols, _, _ = train_and_evaluate(sample_df)
        pred = predict_future(model, sample_df, feature_cols)
        assert pred['predicted_ili'].iloc[0] >= 0

    def test_interval_lower_not_exceed_upper(self, sample_df):
        model, feature_cols, _, _ = train_and_evaluate(sample_df)
        pred = predict_future(model, sample_df, feature_cols)
        assert pred['ci_lower'].iloc[0] <= pred['ci_upper'].iloc[0]

    def test_next_week_date_is_correct(self, sample_df):
        model, feature_cols, _, _ = train_and_evaluate(sample_df)
        pred = predict_future(model, sample_df, feature_cols)
        expected_date = sample_df['week_start'].max() + timedelta(weeks=1)
        assert pred['week_start'].iloc[0] == expected_date


# ---------------------------------------------------------------------------
# check_alerts
# ---------------------------------------------------------------------------

class TestCheckAlerts:
    def test_no_alert_when_below_threshold(self):
        prediction = pd.DataFrame([{'predicted_ili': 100.0, 'week_start': datetime(2024, 1, 1)}])
        alert, threshold = check_alerts(prediction, recent_avg=200.0, threshold_mult=1.5)
        assert alert is None
        assert threshold == 300.0

    def test_alert_when_above_threshold(self):
        prediction = pd.DataFrame([{'predicted_ili': 500.0, 'week_start': datetime(2024, 1, 1)}])
        alert, threshold = check_alerts(prediction, recent_avg=200.0, threshold_mult=1.5)
        assert alert is not None
        assert alert['predicted'] == 500.0
        assert alert['excess'] == 200.0


# ---------------------------------------------------------------------------
# should_send_alert
# ---------------------------------------------------------------------------

class TestShouldSendAlert:
    def test_first_alert_always_sent(self, tmp_path, monkeypatch):
        monkeypatch.setattr('cdc_flu_pipeline.ALERT_COOLDOWN_DAYS', 6)
        with patch('cdc_flu_pipeline.ALERT_COOLDOWN_DAYS', 6):
            out_dir = tmp_path / 'output'
            out_dir.mkdir()
            monkeypatch.chdir(tmp_path)

            alert = {'week': datetime(2024, 3, 15), 'predicted': 500, 'threshold': 300}
            assert should_send_alert(alert) is True

    def test_cooldown_suppresses_repeat_alert(self, tmp_path, monkeypatch):
        monkeypatch.setattr('cdc_flu_pipeline.ALERT_COOLDOWN_DAYS', 6)
        out_dir = tmp_path / 'output'
        out_dir.mkdir()
        # Write a fake last_alert.txt dated today
        with open(out_dir / 'last_alert.txt', 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d') + '\n')
        monkeypatch.chdir(tmp_path)

        with patch('cdc_flu_pipeline.ALERT_COOLDOWN_DAYS', 6):
            alert = {'week': datetime(2024, 3, 15), 'predicted': 500, 'threshold': 300}
            assert should_send_alert(alert) is False


# ---------------------------------------------------------------------------
# create_plot
# ---------------------------------------------------------------------------

class TestCreatePlot:
    def test_returns_bytesio_buffer(self, sample_df):
        prediction = pd.DataFrame([{
            'week_start': sample_df['week_start'].max() + timedelta(weeks=1),
            'predicted_ili': 1500.0,
            'ci_lower': 1200.0,
            'ci_upper': 1800.0,
        }])
        buf = create_plot(sample_df, prediction, threshold=1200.0)
        assert isinstance(buf, io.BytesIO)
        assert buf.getbuffer().nbytes > 0

    def test_plot_marker_when_exceeds_threshold(self, sample_df):
        prediction = pd.DataFrame([{
            'week_start': sample_df['week_start'].max() + timedelta(weeks=1),
            'predicted_ili': 2000.0,
            'ci_lower': 1500.0,
            'ci_upper': 2500.0,
        }])
        buf = create_plot(sample_df, prediction, threshold=1000.0)
        assert isinstance(buf, io.BytesIO)


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------

class TestSaveArtifacts:
    def test_creates_output_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        prediction = pd.DataFrame([{
            'week_start': datetime(2024, 3, 18),
            'predicted_ili': 1500.0,
            'ci_lower': 1200.0,
            'ci_upper': 1800.0,
        }])
        buf = io.BytesIO(b'fake-png-data')

        result = save_artifacts(prediction, buf)
        assert result is True
        assert (tmp_path / 'output' / 'prediction.csv').exists()
        assert (tmp_path / 'output' / 'forecast_plot.png').exists()
        assert (tmp_path / 'output' / 'summary.txt').exists()

        # Verify CSV content
        saved = pd.read_csv(tmp_path / 'output' / 'prediction.csv')
        assert saved['predicted_ili'].iloc[0] == 1500.0


# ---------------------------------------------------------------------------
# send_email (unit; no real SMTP)
# ---------------------------------------------------------------------------

class TestSendEmail:
    def test_skips_when_not_configured(self, monkeypatch):
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_FROM', None)
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_PASSWORD', None)
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_TO', [''])

        alert = {'week': datetime(2024, 3, 18), 'predicted': 500,
                 'threshold': 300, 'excess': 200, 'recent_avg': 250}
        prediction = pd.DataFrame([{
            'week_start': datetime(2024, 3, 18),
            'predicted_ili': 500.0,
            'ci_lower': 400.0,
            'ci_upper': 600.0,
        }])
        buf = io.BytesIO(b'fake-png')
        result = send_email(alert, prediction, buf, 300.0)
        assert result is False

    def test_sends_when_configured(self, monkeypatch):
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_FROM', 'test@example.com')
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_PASSWORD', 'password')
        monkeypatch.setattr('cdc_flu_pipeline.EMAIL_TO', ['to@example.com'])
        monkeypatch.setattr('cdc_flu_pipeline.SMTP_SERVER', 'localhost')
        monkeypatch.setattr('cdc_flu_pipeline.SMTP_PORT', 587)

        alert = {'week': datetime(2024, 3, 18), 'predicted': 500,
                 'threshold': 300, 'excess': 200, 'recent_avg': 250}
        prediction = pd.DataFrame([{
            'week_start': datetime(2024, 3, 18),
            'predicted_ili': 500.0,
            'ci_lower': 400.0,
            'ci_upper': 600.0,
        }])
        # Minimal valid PNG bytes (1×1 red pixel)
        buf = io.BytesIO(bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ]))

        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = send_email(alert, prediction, buf, 300.0)
            assert result is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_cdc_data (integration-style with mocking)
# ---------------------------------------------------------------------------

class TestFetchCdcData:
    def test_fetches_and_parses_delphi_data(self, monkeypatch):
        """Delphi API response: {"result": 1, "epidata": [...]}."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": 1,
            "epidata": [
                {
                    "week_start": "2024-01-07",
                    "epiweek": 202401,
                    "num_ili": 1200,
                    "num_patients": 24000,
                    "ili": 5.0,
                },
                {
                    "week_start": "2024-01-14",
                    "epiweek": 202402,
                    "num_ili": 1350,
                    "num_patients": 25000,
                    "ili": 5.4,
                },
            ],
        }

        with patch("requests.get", return_value=mock_response) as mock_get:
            df = fetch_cdc_data()
            mock_get.assert_called_once()
            assert len(df) == 2
            assert df["ilitotal"].iloc[0] == 1200.0
            assert df["total_patients"].iloc[0] == 24000.0
            assert df["percent_ili"].iloc[0] == 5.0
            assert df["week"].iloc[0] == 202401
            assert isinstance(df["week_start"].iloc[0], pd.Timestamp)

    def test_raises_on_api_error_result(self, monkeypatch):
        """When result != 1 the function should raise ValueError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": -1,
            "message": "bad request",
            "epidata": [],
        }

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="bad request"):
                fetch_cdc_data()

    def test_raises_on_empty_epidata(self, monkeypatch):
        """When epidata is an empty list the function should raise ValueError."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": 1,
            "epidata": [],
        }

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="No data returned"):
                fetch_cdc_data()

    def test_accepts_state_abbr_parameter(self, monkeypatch):
        """When state_abbr is passed, it should be used directly (lowercased)."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": 1,
            "epidata": [
                {
                    "week_start": "2024-01-07",
                    "epiweek": 202401,
                    "num_ili": 1200,
                    "num_patients": 24000,
                    "ili": 5.0,
                },
            ],
        }

        with patch("requests.get", return_value=mock_response) as mock_get:
            df = fetch_cdc_data(state_abbr="TX")
            call_args = mock_get.call_args
            # Verify the regions param uses the provided abbreviation (lowercased)
            assert call_args[1]["params"]["regions"] == "tx"


# ---------------------------------------------------------------------------
# Smoke test: full pipeline doesn't crash
# ---------------------------------------------------------------------------

def test_run_pipeline_no_crash(sample_df, monkeypatch, tmp_path):
    """Ensure run_pipeline() completes without raising."""
    monkeypatch.chdir(tmp_path)

    # Disable email
    monkeypatch.setattr('cdc_flu_pipeline.EMAIL_FROM', None)
    monkeypatch.setattr('cdc_flu_pipeline.EMAIL_PASSWORD', None)
    monkeypatch.setattr('cdc_flu_pipeline.EMAIL_TO', [''])

    with patch('cdc_flu_pipeline.fetch_cdc_data', return_value=sample_df):
        from cdc_flu_pipeline import run_pipeline
        exit_code = run_pipeline()
        assert exit_code == 0

        # Verify artifacts were created
        assert (tmp_path / 'output' / 'prediction.csv').exists()
        assert (tmp_path / 'output' / 'forecast_plot.png').exists()
        assert (tmp_path / 'output' / 'summary.txt').exists()
