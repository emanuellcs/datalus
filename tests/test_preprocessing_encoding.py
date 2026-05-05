import json

import numpy as np
import polars as pl

from datalus.encoding import TabularEncoder
from datalus.preprocessing import ZeroShotPreprocessor


def test_lazy_preprocessing_and_reversible_encoding(tmp_path):
    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed.parquet"
    schema_path = tmp_path / "schema_config.json"
    frame = pl.DataFrame(
        {
            "record_id": [f"id-{idx}" for idx in range(20)],
            "age": list(range(20)),
            "sex": ["M", "F"] * 10,
            "target": [0, 1] * 10,
        }
    )
    frame.write_csv(raw)

    prep = ZeroShotPreprocessor(sample_size=20, target_column="target")
    prep.fit_transform_to_parquet(raw, processed, schema_path)
    schema = json.loads(schema_path.read_text())

    assert schema["record_id"]["encoding_strategy"] == "DROP"
    assert processed.exists()

    processed_frame = pl.read_parquet(processed)
    encoder = TabularEncoder(schema).fit(processed_frame)
    encoded = encoder.transform(processed_frame)
    decoded = encoder.inverse_transform(encoded.x_num, encoded.x_cat)

    assert encoded.x_num is not None
    assert encoded.x_cat is not None
    assert np.isfinite(encoded.x_num).all()
    assert len(decoded) == len(processed_frame)
