import tomllib

with open("pyproject.toml", "rb") as f:
    toml_data = tomllib.load(f)
    app_version = toml_data["project"]["version"]
