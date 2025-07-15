from flask import Flask, render_template, request, redirect, url_for
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = request.form.get('n', type=int)
        if n is None:
            return render_template('formulario.html', error="No se recibió el número de incógnitas.")
        return redirect(url_for('sistema', n=n))
    return render_template('formulario.html')

@app.route('/sistema', methods=['GET', 'POST'])
def sistema():
    if request.method == 'POST':
        n = request.form.get('n', type=int)
        if n is None:
            return render_template('formulario.html', error="No se recibió el número de incógnitas.")
        
        form_elements = [
            (f"Coeficientes de la ecuación {i+1} (separados por espacio):", f"eq{i}")
            for i in range(n)
        ]
        return render_template('sistema.html', n=n, form_elements=form_elements)
    
    n = request.args.get('n', type=int)
    if n is None:
        return render_template('formulario.html', error="No se recibió el número de incógnitas.")
    
    form_elements = [
        (f"Coeficientes de la ecuación {i+1} (separados por espacio):", f"eq{i}")
        for i in range(n)
    ]
    return render_template('sistema.html', n=n, form_elements=form_elements)

@app.route('/resolver', methods=['POST'])
def resolver():
    n = request.form.get('n', type=int)
    if n is None:
        return render_template('formulario.html', error="No se recibió el número de incógnitas.")
    
    A = []
    b = []
    
    for i in range(n):
        eq = request.form.get(f'eq{i}')
        A.append(list(map(float, eq.split())))
    
    b = list(map(float, request.form['b'].split()))
    
    A = np.array(A)
    b = np.array(b)

    try:
        solucion = np.linalg.solve(A, b)
        resultado = [f"x{i+1} = {solucion[i]}" for i in range(n)]
        return render_template('resultado.html', resultado=resultado)
    except np.linalg.LinAlgError as e:
        return render_template('formulario.html', error=f"El sistema no tiene solución única: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
