import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import FloatSlider, IntSlider, Checkbox, Label, VBox, GridBox, Layout, interactive_output, Button
from IPython.display import display

# --- Parámetros del sistema (modelo de planta de primer orden para humedad) ---
K_hum = 1.0        # Ganancia del sistema de humedad
tau_hum = 10.0     # Constante de tiempo del sistema de humedad
HR_amb_base = 60 # Humedad ambiente base (%)

# --- Configuración inicial de la simulación ---
t = np.linspace(0, 100, 1000)
dt = t[1] - t[0]

# --- Función de simulación del controlador Proporcional (P) para humedad ---
def simulate_proportional_humidity(Kp_c, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion,fl_ajustar_controlador,cota_error):
    HR = np.zeros_like(t)         # Humedad relativa
    e = np.zeros_like(t)          # Error
    output = np.zeros_like(t)     # Señal de control (salida del controlador P)

    P_term = np.zeros_like(t)     # Solo término Proporcional
    Kp_ajustado=np.zeros_like(t) 
    Kp=Kp_c
    HR_amb_values = np.zeros_like(t)
    HR_amb_base = HR_inicial
    if Kp==0:
        HR_ref=HR_inicial
    
    # Si la perturbación no está habilitada
    if not fl_perturbacion:
        HR_amb_perturb = HR_amb_base # La perturbación de humedad es la base
        # Set perturbation range to effectively zero duration
        perturbation_start = 0
        perturbation_end = 0       

    for i in range(len(t)):
        # Perturbación de humedad ambiente
        if perturbation_start <= t[i] <= perturbation_end:
            HR_amb = HR_amb_perturb
        else:
            HR_amb = HR_amb_base
        HR_amb_values[i] = HR_amb # Guardar HR_amb para graficar

        if i == 0:
            HR[i] = HR_inicial # Usamos HR_inicial para el primer punto
            e[i] = HR_ref - HR[i]
            continue

        e[i] = HR_ref - HR[i-1] # Error actual

        

        # Ajuste del controlador
        if fl_ajustar_controlador and Kp!=0:
            lvl_e=cota_error/5
            if abs(e[i])<=lvl_e:
                Kp=Kp_c*6
            elif abs(e[i])<=lvl_e*2:
                Kp=Kp_c*5
            elif abs(e[i])<=lvl_e*3:
                Kp=Kp_c*4
            elif abs(e[i])<=lvl_e*4:
                Kp=Kp_c*3
            elif abs(e[i])<=lvl_e*5:
                Kp=Kp_c*2
            else:
                Kp=Kp_c

        Kp_ajustado[i]=Kp
        
        # Componente Proporcional (P)
        P_term[i] = Kp * e[i]
        
        # Señal de control total (solo P)
        output[i] = P_term[i]

        # Modelo de la planta (respuesta a la señal de control y perturbación en HR_amb)
        dHRdt = (K_hum * output[i] - (HR[i-1] - HR_amb)) / tau_hum
        HR[i] = HR[i-1] + dHRdt * dt
    
    return HR, P_term, output, HR_amb_values, e, Kp_ajustado

# --- Función para actualizar el gráfico ---
def update_plot(Kp, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion, rango_error,fl_ajustar_controlador):

    if HR_amb_perturb==HR_ref:
        fl_perturbacion=False

    HR, P_term, output, HR_amb_values, s_error, Kp_ajustado = simulate_proportional_humidity(Kp, HR_ref, HR_inicial, perturbation_start, perturbation_end, HR_amb_perturb, fl_perturbacion,fl_ajustar_controlador,rango_error)

    error_max=HR_ref+rango_error
    error_min=HR_ref-rango_error
    
    fig = make_subplots(rows=4, cols=1,
                        shared_xaxes=False,
                        vertical_spacing=0.08,
                        subplot_titles=(
                            "Respuesta del sistema de Humedad (Control Proporcional)",
                            "Señal de control (output)",
                            "Error",
                            "Kp"
                        ))

    # Subplot 1: Humedad Relativa
    fig.add_trace(go.Scatter(x=t, y=HR, mode='lines', name='Humedad Actual (HR)', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[HR_ref]*len(t), mode='lines', name='Valor Nominal',
                             line=dict(dash='dash', color='red')), row=1, col=1)
    
    # Add acceptable range lines
    fig.add_trace(go.Scatter(x=t, y=[error_max]*len(t), mode='lines', name='Límite Superior Aceptable (55%)',
                             line=dict(dash='dot', color='green')), row=1, col=1) # Changed to dot for consistency
    fig.add_trace(go.Scatter(x=t, y=[error_min]*len(t), mode='lines', name='Límite Inferior Aceptable (45%)',
                             line=dict(dash='dot', color='green')), row=1, col=1) # Changed to dot for consistency

    lim_inf_y_g1=30
    lim_sup_y_g1=70
    fig.update_layout(yaxis1=dict(
        range=[lim_inf_y_g1, lim_sup_y_g1],
        tickvals=[30, 35, 40, 45, 50, 55, 60, 65, 70],
        ticktext=["30", "35", "40", "45", "50", "55", "60", "65", "70"]
    ))


    # Detectar franjas de falla (Temperatura por fuera del rango de error)
 
    franjas = []
    en_fr = False
    inicio = None
    
    for i in range(len(HR)):
        if HR[i] > error_max:
            if not en_fr:
                en_fr = True
                inicio = t[i]
        elif HR[i] < error_min:
            if not en_fr:
                en_fr = True
                inicio = t[i]
        else:
            if en_fr:
                en_fr = False
                fin = t[i]
                franjas.append((inicio, fin))

    # Captura el último tramo si termina en la última posición
    if en_fr:
        franjas.append((inicio, t[-1]))
    
    # Agregar las franjas de falla
    for start, end in franjas:
        fig.add_shape(
            type="rect",
            xref="x1", yref="y1",  # paper en y para cubrir todo el eje Y
            x0=start, x1=end,
            y0=lim_inf_y_g1, y1=lim_sup_y_g1,
            fillcolor="rgba(255, 0, 0, 0.2)",  # rojo transparente
            line_width=0,
            layer="below"
        )

    

    # Subplot 2: Señal de control (output)
    fig.add_trace(go.Scatter(x=t, y=output, mode='lines', name='Señal de Control (P)', line=dict(color='brown')), row=2, col=1)

    # Subplot 3: Error
    fig.add_trace(go.Scatter(x=t, y=s_error, mode='lines', name='Error', line=dict(color='grey')), row=3, col=1)
    
    # Subplot 4: Kp
    fig.add_trace(go.Scatter(x=t, y=Kp_ajustado, mode='lines', name='Kp', line=dict(color='black')), row=4, col=1)
    
    fig.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="x unified",
        title_text="Simulación de Control Proporcional (P) de Humedad",
        title_x=0.5
    )

    fig.update_yaxes(title_text="Humedad Relativa (%)", row=1, col=1)
    fig.update_yaxes(title_text="Señal de Control", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=3, col=1)
    fig.update_yaxes(title_text="Kp", row=4, col=1)
    
    fig.update_xaxes(title_text="Tiempo (s)", row=4, col=1)

    fig.show()

# --- Crear los controles interactivos con ipywidgets ---

#Sintonizacion del controlador
Kp_hum_inicial=2

Kp_slider_hum = FloatSlider(min=0.0, max=10.0, step=0.1, value=Kp_hum_inicial, description='Kp:')
HR_ref_slider = FloatSlider(min=30.0, max=70.0, step=1.0, value=50.0, description='HR_ref (%):')
HR_inicial_slider = FloatSlider(min=30.0, max=70.0, step=1.0, value=46.0, description='HR Inicial (%):') # Nuevo slider para HR inicial

# Define initial min/max for sliders to avoid immediate conflicts
initial_perturb_start_value_hum = 30
initial_perturb_end_value_hum = 50

perturbation_start_slider_hum = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_start_value_hum,
    description='Inicio Perturbación (s):', continuous_update=True
)
perturbation_end_slider_hum = IntSlider(
    min=0, max=100, step=1, value=initial_perturb_end_value_hum,
    description='Fin Perturbación (s):', continuous_update=True
)
HR_amb_perturb_slider = FloatSlider(min=10.0, max=90.0, step=1.0, value=75.0, description='HR_amb Perturbación (%):')
chk_perturbacion_hum = Checkbox(value=False, description='Perturbación', disabled=False, indent=False)
rango_error = IntSlider(min=0, max=20, step=1, value=5, description='Rango Error (+/-):', continuous_update=True)
boton_reset_controlador = Button(description="Reiniciar Controlador Proporcional", button_style="")
chk_ajustar_Kp = Checkbox(value=False, description='Ajustar controlador', disabled=False, indent=False,continuous_update=True)

# Función que se ejecutará al hacer clic
def reset_KP(b):
    Kp_slider_hum.value=Kp_hum_inicial

# Asignar la función al evento click
boton_reset_controlador.on_click(reset_KP)

# --- Observadores para los sliders de perturbación (Lógica Corregida y Reforzada para Humedad) ---
def on_perturbation_start_change_hum(change):
    new_start_value = change['new']
    
    # Ensure end_slider's value is never less than new_start_value
    if perturbation_end_slider_hum.value < new_start_value:
        perturbation_end_slider_hum.value = new_start_value
    # Then set the min limit for the end slider
    perturbation_end_slider_hum.min = new_start_value


def on_perturbation_end_change_hum(change):
    new_end_value = change['new']

    # Ensure start_slider's value is never greater than new_end_value
    if perturbation_start_slider_hum.value > new_end_value:
        perturbation_start_slider_hum.value = new_end_value
    # Then set the max limit for the start slider
    perturbation_start_slider_hum.max = new_end_value


# Attach observers for humidity sliders
perturbation_start_slider_hum.observe(on_perturbation_start_change_hum, names='value')
perturbation_end_slider_hum.observe(on_perturbation_end_change_hum, names='value')


# --- Habilitar o deshabilitar sliders de perturbación para humedad ---
def toggle_perturbation_sliders_hum(change):
    disable_sliders = not change['new']
    perturbation_start_slider_hum.disabled = disable_sliders
    perturbation_end_slider_hum.disabled = disable_sliders
    HR_amb_perturb_slider.disabled = disable_sliders

    # IMPORTANT: When enabling, ensure the sliders are in a consistent state
    if not disable_sliders:
        # If start is greater than end, set end to start (or vice-versa)
        if perturbation_start_slider_hum.value > perturbation_end_slider_hum.value:
            perturbation_end_slider_hum.value = perturbation_start_slider_hum.value
        # Also ensure max/min bounds are correct after potential value changes
        # Triggering the observers with current values ensures bounds are synced
        on_perturbation_start_change_hum({'new': perturbation_start_slider_hum.value})
        on_perturbation_end_change_hum({'new': perturbation_end_slider_hum.value})


chk_perturbacion_hum.observe(toggle_perturbation_sliders_hum, names='value')

# Trigger initial state for perturbation sliders based on checkbox value
# Call this *after* setting up observers to ensure correct initial synchronization
toggle_perturbation_sliders_hum({'new': chk_perturbacion_hum.value})

# Manually trigger initial synchronization of slider bounds (redundant but safe)
# This handles the case where initial values might already be invalid BEFORE any user interaction
# if perturbation_start_slider_hum.value > perturbation_end_slider_hum.value:
#     perturbation_end_slider_hum.value = perturbation_start_slider_hum.value
on_perturbation_start_change_hum({'new': perturbation_start_slider_hum.value})
on_perturbation_end_change_hum({'new': perturbation_end_slider_hum.value})


# Columnas para los controles
c1_hum = VBox([Label("🎛️ Control Proporcional"), Kp_slider_hum, boton_reset_controlador, chk_ajustar_Kp])
c2_hum = VBox([Label("💧 Perturbación Humedad"), perturbation_start_slider_hum, perturbation_end_slider_hum, HR_amb_perturb_slider, chk_perturbacion_hum])
c3_hum = VBox([Label("⚙️ Configuraciones Adicionales"), HR_ref_slider, HR_inicial_slider, rango_error])

# Layout para las columnas de los controles
layoutControles_hum = GridBox(
    children=[c1_hum, c2_hum, c3_hum],
    layout=Layout(
        grid_template_columns="repeat(3, auto)",
        grid_gap="10px",
        width="100%",
        justify_items="flex-start",
    )
)

interactive_plot_hum = interactive_output(update_plot, {
    'Kp': Kp_slider_hum,
    'HR_ref': HR_ref_slider,
    'HR_inicial': HR_inicial_slider,
    'perturbation_start': perturbation_start_slider_hum,
    'perturbation_end': perturbation_end_slider_hum,
    'HR_amb_perturb': HR_amb_perturb_slider,
    'fl_perturbacion': chk_perturbacion_hum,
    'rango_error': rango_error,
    'fl_ajustar_controlador': chk_ajustar_Kp
})

# Mostrar controles y gráfico juntos
display(VBox([layoutControles_hum, interactive_plot_hum]))