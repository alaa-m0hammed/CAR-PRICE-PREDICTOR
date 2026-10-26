import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


# Vercel does not serve Flask's static folder, so the stylesheet is served
# by an explicit route. This works the same locally and on Vercel.
@app.route("/style.css")
def stylesheet():
    return send_from_directory(
        os.path.join(BASE_DIR, "static"), "style.css", mimetype="text/css"
    )


with open(os.path.join(BASE_DIR, "LinearRegression.pkl"), "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Build the form options and validation limits from the cleaned dataset, so
# the dropdowns and ranges always match what the model was trained on.
# ---------------------------------------------------------------------------
FEATURES = ["name", "company", "year", "kms_driven", "fuel_type"]

car = pd.read_csv(os.path.join(BASE_DIR, "cleaned_quikr_car.csv"))
car = car.dropna(subset=FEATURES).copy()
for col in ("name", "company", "fuel_type"):
    car[col] = car[col].astype(str).str.strip()
car["year"] = car["year"].astype(int)
car["kms_driven"] = car["kms_driven"].astype(int)

# The model was fitted on a train split, so it only knows some of the car
# names in the CSV. Offer only names the model has actually learned; any
# other name would be silently ignored by the encoder and give a less
# specific estimate.
try:
    _pipe = getattr(model, "regressor_", model)  # unwrap TransformedTargetRegressor
    _ohe = _pipe.named_steps["columntransformer"].named_transformers_["onehotencoder"]
    car = car[car["name"].isin(set(_ohe.categories_[0]))]
except (AttributeError, KeyError, IndexError):
    pass  # unexpected pipeline layout: fall back to every name in the CSV

MODELS_BY_COMPANY = {
    company: sorted(group["name"].unique().tolist())
    for company, group in car.groupby("company")
}
COMPANIES = sorted(MODELS_BY_COMPANY)
FUEL_TYPES = sorted(car["fuel_type"].unique().tolist())
YEAR_MIN = int(car["year"].min())
YEAR_MAX = int(car["year"].max())
KMS_MAX = int(car["kms_driven"].max())

CATALOG = {
    "companies": COMPANIES,
    "models": MODELS_BY_COMPANY,
    "fuel_types": FUEL_TYPES,
    "year_min": YEAR_MIN,
    "year_max": YEAR_MAX,
    "kms_max": KMS_MAX,
}


def validate(form):
    """Return (clean_values, errors). Errors is a dict of field -> message."""
    values, errors = {}, {}

    company = (form.get("company") or "").strip()
    if company not in MODELS_BY_COMPANY:
        errors["company"] = "Choose a manufacturer from the list."
    values["company"] = company

    name = (form.get("name") or "").strip()
    if "company" not in errors and name not in MODELS_BY_COMPANY[company]:
        errors["name"] = "Choose a car model that belongs to the selected manufacturer."
    values["name"] = name

    fuel_type = (form.get("fuel_type") or "").strip()
    if fuel_type not in FUEL_TYPES:
        errors["fuel_type"] = "Choose a fuel type from the list."
    values["fuel_type"] = fuel_type

    try:
        year = int(form.get("year", ""))
        if not YEAR_MIN <= year <= YEAR_MAX:
            raise ValueError
    except ValueError:
        errors["year"] = f"Enter a year between {YEAR_MIN} and {YEAR_MAX}."
        year = form.get("year", "")
    values["year"] = year

    try:
        kms = int(form.get("kms_driven", ""))
        if not 0 <= kms <= KMS_MAX:
            raise ValueError
    except ValueError:
        errors["kms_driven"] = f"Enter kilometers driven between 0 and {KMS_MAX:,}."
        kms = form.get("kms_driven", "")
    values["kms_driven"] = kms

    return values, errors


def predict_price(values):
    row = pd.DataFrame([{k: values[k] for k in FEATURES}], columns=FEATURES)
    return float(model.predict(row)[0])


@app.route("/")
def index():
    return render_template(
        "index.html", catalog=CATALOG, values={}, errors={}, prediction=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    wants_json = request.headers.get("X-Requested-With") == "fetch"
    values, errors = validate(request.form)

    if errors:
        if wants_json:
            return jsonify(ok=False, errors=errors), 400
        return (
            render_template(
                "index.html",
                catalog=CATALOG,
                values=values,
                errors=errors,
                prediction=None,
            ),
            400,
        )

    price = predict_price(values)
    prediction = {"value": round(price), "reliable": price > 0}

    if wants_json:
        return jsonify(ok=True, prediction=prediction)
    return render_template(
        "index.html",
        catalog=CATALOG,
        values=values,
        errors={},
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
