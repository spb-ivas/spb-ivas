# Earlier SPBOM implementation

The `release/original` directory preserves the historical Python OLS, data-preparation and Dash sources. The `release/runtime` directory holds the adapters used in the earlier comparison. These files document the historical fixed-coefficient example. The SoftwareX demonstration uses the fitted-model commands in the repository README.

The earlier launcher accepts `python release/run_spbom.py --legacy --help` from the repository root. Historical runs require the earlier package versions in `legacy/release/requirements.txt`, shapefile sidecars and additional input tables. The two processed downloads implement the current county workflow.
