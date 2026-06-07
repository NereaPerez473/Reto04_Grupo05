import joblib # o usar pickle si da error

# Cargar el modelo--> Alternar entre eolico/solar segun interes
modelo_eolico = joblib.load("modelo_solar.pkl")

# Intentar extraer los nombres de las características esperadas
if hasattr(modelo_eolico, "feature_names_in_"):
    print("El modelo solar espera exactamente estas columnas:")
    print(modelo_eolico.feature_names_in_)
else:
    print("El modelo no guardó los nombres de las columnas. Usaremos el orden de las diapositivas.")