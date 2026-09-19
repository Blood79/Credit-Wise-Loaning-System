# Architecture

CSV or synthetic demo → validation and target mapping → ColumnTransformer → classifier → metrics → Streamlit UI.

The core design keeps data preparation and model construction outside the Streamlit presentation layer, which makes the workflow testable and reusable.
