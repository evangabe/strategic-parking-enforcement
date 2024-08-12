from dash import Dash, html

app = Dash()

app.layout = [
    html.Div(children='City of Los Angeles: Proposing a Strategy for Optimizing Parking Enforcement Deployment', style={'fontSize': 32})
]

if __name__ == '__main__':
    app.run(debug=True)