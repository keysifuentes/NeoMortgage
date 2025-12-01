import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ===========================================
# 1. FUNCIONES FINANCIERAS
# ===========================================

def pago_mensual_hipoteca(monto, tasa_anual, plazo_anos):
    """
    Calcula pago mensual aproximado de una hipoteca con tasa fija.
    tasa_anual en %, plazo_anos en años.
    """
    tasa_mensual = (tasa_anual / 100) / 12
    n_meses = plazo_anos * 12

    if n_meses <= 0:
        raise ValueError("plazo_anos debe ser positivo")

    if tasa_mensual <= 0:
        # Caso raro: tasa 0
        return monto / n_meses

    pago = monto * tasa_mensual / (1 - (1 + tasa_mensual) ** (-n_meses))
    return pago


def generar_tabla_amortizacion(monto, tasa_anual, plazo_anos):
    """
    Genera una tabla de amortización agregada por AÑO.
    Devuelve:
      - lista de dicts por año
      - total_intereses_pagados
      - total_pagado (capital + intereses)
    """
    tasa_mensual = (tasa_anual / 100) / 12
    n_meses = plazo_anos * 12
    pago_mensual = pago_mensual_hipoteca(monto, tasa_anual, plazo_anos)

    saldo = monto
    tabla_anual = []
    total_intereses = 0.0
    total_pagado = 0.0

    mes_global = 0
    for anio in range(1, plazo_anos + 1):
        saldo_inicial_anio = saldo
        interes_anual = 0.0
        capital_anual = 0.0
        pago_anual = 0.0

        for _ in range(12):
            if mes_global >= n_meses or saldo <= 0:
                break

            interes_mes = saldo * tasa_mensual
            capital_mes = pago_mensual - interes_mes
            if capital_mes < 0:
                capital_mes = 0.0

            saldo = saldo - capital_mes
            if saldo < 0:
                saldo = 0.0

            interes_anual += interes_mes
            capital_anual += capital_mes
            pago_anual += pago_mensual

            mes_global += 1

        total_intereses += interes_anual
        total_pagado += pago_anual

        tabla_anual.append({
            "anio": anio,
            "saldo_inicial": round(saldo_inicial_anio, 2),
            "capital_pagado": round(capital_anual, 2),
            "interes_pagado": round(interes_anual, 2),
            "pago_total": round(pago_anual, 2),
            "saldo_final": round(saldo, 2),
        })

        if mes_global >= n_meses or saldo <= 0:
            break

    return tabla_anual, round(total_intereses, 2), round(total_pagado, 2)


def crear_modelo_credito_hipotecario(n=10000, random_state=42):
    np.random.seed(random_state)

    # Ingreso mensual del cliente
    ingreso_mensual = np.random.normal(loc=25000, scale=8000, size=n)
    ingreso_mensual = np.clip(ingreso_mensual, 6000, 150000)

    # Valor de la vivienda
    valor_vivienda = np.random.normal(loc=1500000, scale=400000, size=n)
    valor_vivienda = np.clip(valor_vivienda, 300000, 10000000)

    # LTV simulado
    ltv_sim = np.random.uniform(0.5, 0.95, size=n)
    monto_credito = valor_vivienda * ltv_sim

    # Plazo en años
    plazo_anos = np.random.choice([10, 15, 20, 25, 30], size=n,
                                  p=[0.1, 0.2, 0.3, 0.25, 0.15])

    # Tasa hipotecaria simulada
    tasa_credito = np.random.normal(loc=10.5, scale=1.5, size=n)
    tasa_credito = np.clip(tasa_credito, 7.0, 15.0)

    # Edad solicitante
    edad = np.random.normal(loc=37, scale=8, size=n)
    edad = np.clip(edad, 22, 75)

    # Antigüedad laboral
    antiguedad_laboral = np.random.exponential(scale=6, size=n)
    antiguedad_laboral = np.clip(antiguedad_laboral, 0, 40)

    # Otras deudas mensuales
    otras_deudas_mensuales = np.random.normal(loc=4000, scale=2500, size=n)
    otras_deudas_mensuales = np.clip(otras_deudas_mensuales, 0, 40000)

    # Tipo vivienda
    tipo_vivienda = np.random.choice(
        ['interes_social', 'media', 'residencial', 'lujo'],
        size=n,
        p=[0.3, 0.4, 0.25, 0.05]
    )

    # Riesgo ubicación
    ubicacion_riesgo = np.random.choice(
        ['bajo', 'medio', 'alto'],
        size=n,
        p=[0.4, 0.4, 0.2]
    )

    # Historial
    num_atrasos_12m = np.random.poisson(lam=0.3, size=n)
    score_interno_prev = np.random.normal(loc=680, scale=40, size=n)
    score_interno_prev = np.clip(score_interno_prev, 500, 850)

    # Derivadas
    pago_mensual = np.array([
        pago_mensual_hipoteca(monto_credito[i], tasa_credito[i], plazo_anos[i])
        for i in range(n)
    ])

    relacion_pago_ingreso = (pago_mensual + otras_deudas_mensuales) / (ingreso_mensual + 1)
    ltv = monto_credito / (valor_vivienda + 1)

    # Coefs simulados para PD
    beta_0 = -4.0
    beta_ingreso = -0.00002
    beta_valor = -0.0000003
    beta_monto = 0.0000005
    beta_plazo = 0.04
    beta_tasa = 0.12
    beta_edad = -0.01
    beta_antiguedad = -0.03
    beta_otras_deudas = 0.00006
    beta_ltv = 3.0
    beta_rpi = 4.0
    beta_atrasos = 0.45
    beta_score = -0.006

    tipo_vivienda_efectos = {
        'interes_social': 0.10,
        'media': 0.0,
        'residencial': -0.05,
        'lujo': 0.05
    }

    ubicacion_efectos = {
        'bajo': -0.05,
        'medio': 0.0,
        'alto': 0.25
    }

    logit_pd = (
        beta_0
        + beta_ingreso * ingreso_mensual
        + beta_valor * valor_vivienda
        + beta_monto * monto_credito
        + beta_plazo * plazo_anos
        + beta_tasa * tasa_credito
        + beta_edad * edad
        + beta_antiguedad * antiguedad_laboral
        + beta_otras_deudas * otras_deudas_mensuales
        + beta_ltv * ltv
        + beta_rpi * relacion_pago_ingreso
        + beta_atrasos * num_atrasos_12m
        + beta_score * score_interno_prev
        + np.array([tipo_vivienda_efectos[t] for t in tipo_vivienda])
        + np.array([ubicacion_efectos[u] for u in ubicacion_riesgo])
    )

    pd_real = 1 / (1 + np.exp(-logit_pd))
    default = np.random.binomial(n=1, p=pd_real, size=n)

    data = pd.DataFrame({
        'ingreso_mensual': ingreso_mensual,
        'valor_vivienda': valor_vivienda,
        'monto_credito': monto_credito,
        'plazo_anos': plazo_anos,
        'tasa_credito': tasa_credito,
        'edad': edad,
        'antiguedad_laboral': antiguedad_laboral,
        'otras_deudas_mensuales': otras_deudas_mensuales,
        'ltv': ltv,
        'relacion_pago_ingreso': relacion_pago_ingreso,
        'tipo_vivienda': tipo_vivienda,
        'ubicacion_riesgo': ubicacion_riesgo,
        'num_atrasos_12m': num_atrasos_12m,
        'score_interno_prev': score_interno_prev,
        'default': default
    })

    X = data.drop(columns=['default'])
    y = data['default']

    numeric_features = [
        'ingreso_mensual', 'valor_vivienda', 'monto_credito', 'plazo_anos',
        'tasa_credito', 'edad', 'antiguedad_laboral', 'otras_deudas_mensuales',
        'ltv', 'relacion_pago_ingreso', 'num_atrasos_12m', 'score_interno_prev'
    ]

    categorical_features = [
        'tipo_vivienda', 'ubicacion_riesgo'
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    log_reg = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', log_reg)
    ])

    model_pipeline.fit(X_train, y_train)

    # Distribución de PD para la gráfica
    pd_dist = model_pipeline.predict_proba(X)[:, 1]
    pd_mean = float(pd_dist.mean())
    pd_std = float(pd_dist.std() + 1e-6)

    counts, bin_edges = np.histogram(pd_dist, bins=10, range=(0, 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return model_pipeline, counts.tolist(), bin_centers.tolist(), pd_mean, pd_std


modelo, hist_counts, hist_bins, pd_mean, pd_std = crear_modelo_credito_hipotecario()


def clasificar_riesgo(prob_default):
    if prob_default < 0.03:
        return "MUY BAJO"
    elif prob_default < 0.08:
        return "BAJO"
    elif prob_default < 0.15:
        return "MEDIO"
    elif prob_default < 0.30:
        return "ALTO"
    else:
        return "MUY ALTO"


def interpretar_macro_riesgo(tasa_banxico):
    if tasa_banxico < 6:
        return "Entorno expansivo: hipotecas más baratas, mayor demanda pero riesgo de sobreapalancamiento."
    elif tasa_banxico < 10:
        return "Entorno neutral: condiciones relativamente estables para originar hipotecas."
    else:
        return "Entorno restrictivo: hipotecas caras, mayor sensibilidad a atrasos y morosidad."


def calcular_parametros_negocio(monto_credito, pd, ltv, tasa_banxico, tasa_cliente):
    """
    Devuelve dict con:
    - costo_fondeo_pct
    - lgd_pct
    - el_pesos, el_pct
    - tasa_min_rentable_pct
    - margen_credito_pct
    - utilidad_esperada_pesos (por margen)
    - penalizacion_mora_pct, descuento_pronto_pago_pct
    - ing_penalizacion_esperados, descuento_pronto_pago_esperado
    """
    # Supuestos de negocio
    spread_fondeo = 3.0          # Banxico + 3 pp
    margen_objetivo = 3.0        # 3% margen objetivo

    # LGD base según LTV
    if ltv <= 0.8:
        lgd_pct = 0.25
    elif ltv <= 0.9:
        lgd_pct = 0.35
    else:
        lgd_pct = 0.45

    costo_fondeo_pct = tasa_banxico + spread_fondeo      # en %
    ead = monto_credito

    # Pérdida esperada (Expected Loss)
    el_pesos = pd * lgd_pct * ead
    el_pct = el_pesos / (ead + 1e-6)                     # ≈ PD * LGD

    # Tasa mínima rentable
    tasa_min_rentable_pct = costo_fondeo_pct + (el_pct * 100) + margen_objetivo

    # Tasa actual para el cliente
    tasa_cliente_pct = tasa_cliente

    # Costo total (fondeo + riesgo esperado)
    costo_riesgo_total_pct = costo_fondeo_pct + (el_pct * 100)

    # Margen sobre costo total
    margen_credito_pct = tasa_cliente_pct - costo_riesgo_total_pct

    # Utilidad esperada simple por margen (aprox, sin mora/prepago)
    utilidad_esperada_pesos = (margen_credito_pct / 100.0) * ead

    # ----- Penalización por mora y descuento por pronto pago -----
    penalizacion_mora_pct = 3.0            # +3 pp anuales
    descuento_pronto_pago_pct = 1.0        # -1 pp

    # Probabilidades aproximadas según PD
    prob_mora = min(1.0, pd * 2.0)
    prob_prepago = max(0.0, 1.0 - pd * 2.0)

    # Ingreso esperado por penalización (mora)
    ing_penalizacion_esperados = prob_mora * ead * (penalizacion_mora_pct / 100.0) * 0.5

    # Descuento esperado por pronto pago
    descuento_pronto_pago_esperado = prob_prepago * ead * (descuento_pronto_pago_pct / 100.0) * 0.5

    return {
        'costo_fondeo_pct': round(costo_fondeo_pct, 2),
        'lgd_pct': round(lgd_pct * 100, 2),
        'el_pesos': round(el_pesos, 2),
        'el_pct': round(el_pct * 100, 2),
        'tasa_min_rentable_pct': round(tasa_min_rentable_pct, 2),
        'margen_credito_pct': round(margen_credito_pct, 2),
        'utilidad_esperada_pesos': round(utilidad_esperada_pesos, 2),
        'penalizacion_mora_pct': round(penalizacion_mora_pct, 2),
        'descuento_pronto_pago_pct': round(descuento_pronto_pago_pct, 2),
        'ing_penalizacion_esperados': round(ing_penalizacion_esperados, 2),
        'descuento_pronto_pago_esperado': round(descuento_pronto_pago_esperado, 2)
    }


# ===========================================
# 4. APLICACIÓN FLASK (NEOMORTGAGE)
# ===========================================

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    macro_info = None
    tasa_banxico_val = None
    amortizacion_anual = None

    if request.method == 'POST':
        try:
            
            ingreso_mensual = float(request.form['ingreso_mensual'])
            valor_vivienda = float(request.form['valor_vivienda'])
            monto_credito = float(request.form['monto_credito'])
            plazo_anos = int(request.form['plazo_anos'])
            tasa_credito = float(request.form['tasa_credito'])

            edad = int(request.form['edad'])
            antiguedad_laboral = float(request.form['antiguedad_laboral'])
            otras_deudas_mensuales = float(request.form['otras_deudas_mensuales'])

            tipo_vivienda = request.form['tipo_vivienda']
            ubicacion_riesgo = request.form['ubicacion_riesgo']

            num_atrasos_12m = int(request.form['num_atrasos_12m'])
            score_interno_prev = float(request.form['score_interno_prev'])

            tasa_banxico_val = float(request.form['tasa_banxico'])
            macro_info = interpretar_macro_riesgo(tasa_banxico_val)

            # 2. Derivadas
            pago_mensual = pago_mensual_hipoteca(monto_credito, tasa_credito, plazo_anos)
            relacion_pago_ingreso = (pago_mensual + otras_deudas_mensuales) / (ingreso_mensual + 1)
            ltv = monto_credito / (valor_vivienda + 1)

            # 3. Amortización del crédito del cliente
            amortizacion_anual, total_intereses_credito, total_pagado_credito = generar_tabla_amortizacion(
                monto_credito, tasa_credito, plazo_anos
            )

            # 4. Predicción de PD
            df_nuevo = pd.DataFrame({
                'ingreso_mensual': [ingreso_mensual],
                'valor_vivienda': [valor_vivienda],
                'monto_credito': [monto_credito],
                'plazo_anos': [plazo_anos],
                'tasa_credito': [tasa_credito],
                'edad': [edad],
                'antiguedad_laboral': [antiguedad_laboral],
                'otras_deudas_mensuales': [otras_deudas_mensuales],
                'ltv': [ltv],
                'relacion_pago_ingreso': [relacion_pago_ingreso],
                'tipo_vivienda': [tipo_vivienda],
                'ubicacion_riesgo': [ubicacion_riesgo],
                'num_atrasos_12m': [num_atrasos_12m],
                'score_interno_prev': [score_interno_prev]
            })

            proba_default = modelo.predict_proba(df_nuevo)[:, 1][0]
            decision = 1 if proba_default >= 0.5 else 0  # 1 = rechazar

            z_score = (proba_default - pd_mean) / pd_std
            banda_riesgo = clasificar_riesgo(proba_default)

            # 5. Parámetros de negocio
            negocio = calcular_parametros_negocio(
                monto_credito=monto_credito,
                pd=proba_default,
                ltv=ltv,
                tasa_banxico=tasa_banxico_val,
                tasa_cliente=tasa_credito
            )

            # 6. Amortización del fondeo (lo que tú pagas por el dinero)
            _, total_intereses_fondeo, _ = generar_tabla_amortizacion(
                monto_credito, negocio['costo_fondeo_pct'], plazo_anos
            )

            # 7. Vista interna de rentabilidad
            ingresos_intereses_brutos = total_intereses_credito
            costo_intereses_fondeo = total_intereses_fondeo
            perdida_esperada = negocio['el_pesos']

            utilidad_neta_base = ingresos_intereses_brutos - costo_intereses_fondeo - perdida_esperada

            # Ajuste por penalización de mora y descuento por pronto pago
            utilidad_neta_ajustada = (
                utilidad_neta_base
                + negocio['ing_penalizacion_esperados']
                - negocio['descuento_pronto_pago_esperado']
            )

            margen_total_pct = (utilidad_neta_ajustada / (monto_credito + 1e-6)) * 100

            # 8. Armar resultado para el template
            resultado = {
                'prob_default_pct': round(proba_default * 100, 2),
                'prob_default_raw': float(proba_default),
                'decision': 'RECHAZAR' if decision == 1 else 'APROBAR',
                'z_score': round(z_score, 2),
                'banda_riesgo': banda_riesgo,

                'pago_mensual_aprox': round(pago_mensual, 2),
                'relacion_pago_ingreso': round(relacion_pago_ingreso, 2),
                'ltv': round(ltv, 2),
                'tasa_credito': tasa_credito,

                # Negocio base
                'costo_fondeo_pct': negocio['costo_fondeo_pct'],
                'lgd_pct': negocio['lgd_pct'],
                'el_pesos': negocio['el_pesos'],
                'el_pct': negocio['el_pct'],
                'tasa_min_rentable_pct': negocio['tasa_min_rentable_pct'],
                'margen_credito_pct': negocio['margen_credito_pct'],
                'utilidad_esperada_pesos': negocio['utilidad_esperada_pesos'],

                # Políticas de mora / pronto pago
                'penalizacion_mora_pct': negocio['penalizacion_mora_pct'],
                'descuento_pronto_pago_pct': negocio['descuento_pronto_pago_pct'],
                'ing_penalizacion_esperados': negocio['ing_penalizacion_esperados'],
                'descuento_pronto_pago_esperado': negocio['descuento_pronto_pago_esperado'],

                # Amortización cliente
                'total_intereses_credito': total_intereses_credito,
                'total_pagado_credito': total_pagado_credito,

                # Panel interno de rentabilidad ajustada
                'ingresos_intereses_brutos': round(ingresos_intereses_brutos, 2),
                'costo_intereses_fondeo': round(costo_intereses_fondeo, 2),
                'utilidad_neta_pesos': round(utilidad_neta_ajustada, 2),
                'margen_total_pct': round(margen_total_pct, 2)
            }

        except Exception as e:
            resultado = {
                'error': f"Error al procesar la solicitud: {e}"
            }

    return render_template(
        'index.html',
        resultado=resultado,
        macro_info=macro_info,
        tasa_banxico_val=tasa_banxico_val,
        hist_counts=hist_counts,
        hist_bins=hist_bins,
        amortizacion_anual=amortizacion_anual
    )


if __name__ == '__main__':
    app.run(debug=True)
