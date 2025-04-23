from flask import Flask, redirect, url_for, render_template
from codeforpaint import VirtualPainter

app = Flask(__name__)

# Register the blueprint for VirtualPainter (assumed to handle the recognition system)
app.register_blueprint(VirtualPainter, url_prefix="")

if __name__ == "__main__":
    app.run(debug=True)
