from flask import Flask, render_template, request, redirect, url_for
import os
from forecasting import preprocess_data, train_and_forecast

app = Flask(__name__)
UPLOAD_FOLDER = './uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Halaman utama untuk upload file.
    """
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file selected!", 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return redirect(url_for("forecast_options", filename=file.filename))
    return render_template("index.html")

@app.route("/forecast/<filename>", methods=["GET", "POST"])
def forecast_options(filename):
    """
    Halaman opsi forecasting.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        data, columns = preprocess_data(filepath)
        columns = [col for col in columns if col != "total load actual"]  # Hilangkan "total load actual" dari dropdown
    except ValueError as e:
        return render_template("error.html", message=str(e))
    
    if request.method == "POST":
        horizon = request.form["horizon"]
        variable = request.form["variable"]
        return redirect(url_for("forecast_result", filename=filename, horizon=horizon, variable=variable))
    return render_template("forecast_options.html", columns=columns)

@app.route("/forecast/result/<filename>/<horizon>/<variable>")
def forecast_result(filename, horizon, variable):
    """
    Hasil forecasting.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        # Tangani opsi "overall" sebagai forecasting "total load actual"
        plot_html = train_and_forecast(filepath, variable, horizon)
    except Exception as e:
        return render_template("error.html", message=str(e))
    return render_template("result.html", plot_div=plot_html, variable="Overall" if variable == "overall" else variable)

if __name__ == "__main__":
    app.run(debug=True)