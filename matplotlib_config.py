
# Configuración de Matplotlib para evitar warnings de glifos
import matplotlib as mpl

# Establecer opciones globales de matplotlib
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
mpl.rcParams['font.family'] = 'sans-serif'

# Desactivar uso de caracteres Unicode en los títulos
mpl.rcParams['axes.unicode_minus'] = False

print("Configuración de matplotlib ajustada para evitar warnings de glifos")
